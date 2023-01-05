use std::env;

use itertools::Itertools;
use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;

type Note = i32;

// A0 to C8
const NOTES_RANGE: std::ops::Range<i32> = -48..40;

/// Computes the frequency of the given note, where 'note' is an integer representing the number of
/// semitones above (+ve) or below (-ve) A4. Assumes A4 is tuned to 440Hz and notes are distributed
/// according to even temperament (semitones differ by the 12th root of 2).
fn get_note_freq(note: Note) -> f32 {
    440.0_f32 * (note as f32 / 12.0_f32).exp2()
}

fn closest_note(frequency: f32) -> Option<Note> {
    if frequency < get_note_freq(NOTES_RANGE.start + 12)
        || frequency > get_note_freq(NOTES_RANGE.end)
    {
        None
    } else {
        Some((12.0_f32 * (frequency / 440.0_f32).log2()).round() as Note)
    }
}

#[allow(dead_code)]
fn get_note_name(note: Note) -> String {
    // Consider 0 to be the middle C in this context, so our "A above middle C"
    // is now 9
    let c_note = note + 9;

    let index = (c_note + 48) % 12;
    let octave = (c_note + 48) / 12;

    let note_name = match index {
        0 => "C",
        1 => "C♯",
        2 => "D",
        3 => "D♯",
        4 => "E",
        5 => "F",
        6 => "F♯",
        7 => "G",
        8 => "G♯",
        9 => "A",
        10 => "A♯",
        11 => "B",
        _ => panic!("Invalid match"),
    };

    format!("{note_name}{octave}")
}

fn load(filename: &str) -> Result<(creak::AudioInfo, creak::SampleIterator), creak::DecoderError> {
    let decoder = creak::Decoder::open(filename)?;
    Ok((decoder.info(), decoder.into_samples()?))
}

/// Does some preliminary processing on the SampleIterator to make the upcoming
/// work easier:
/// - unwraps the Results for each sample
/// - remove samples from all but the first channel
fn merge_audio(
    info: &creak::AudioInfo,
    samples: creak::SampleIterator,
) -> impl Iterator<Item = f32> + 'static {
    let channels = info.channels();
    samples.into_iter().map(|x| x.unwrap()).step_by(channels)
}

#[allow(dead_code)]
fn print_frequency_spectrum(bins: &[Complex<f32>]) {
    for (index, bin) in bins.iter().enumerate() {
        // figure out the magnitude of the complex number, then print one # per 0.1?
        let magnitude = bin.norm();
        print!("bin {index}: ");

        for _ in 0..magnitude as usize {
            print!("#");
        }

        println!();
    }
}

/// Convert the given samples into the frequency spectrum.
/// Zip the resulting iterator with NOTES_RANGE to get the int/f32 Note/magnitude pair
fn do_fourier_transform(samples: &[f32]) -> impl Iterator<Item = f32> + 'static {
    // make an FFT planner
    let mut real_planner = RealFftPlanner::<f32>::new();

    // create a FFT
    let r2c = real_planner.plan_fft_forward(samples.len());

    // make input and output vectors
    let mut indata = r2c.make_input_vec();
    let mut spectrum = r2c.make_output_vec();

    assert_eq!(spectrum.len(), samples.len() / 2 + 1);

    indata[..].clone_from_slice(samples);

    // Forward transform the input data
    r2c.process(&mut indata, &mut spectrum).unwrap();

    spectrum.into_iter().map(|x| x.norm())
}

fn main() {
    let args = env::args().collect_vec();

    let app_name = &args[0];

    if args.len() < 2 {
        println!("Usage: {app_name} <audio file>");
        std::process::exit(1);
    }

    let song_name = &args[1];

    let (audio_info, samples) = load(song_name).unwrap();

    let samples = merge_audio(&audio_info, samples).collect_vec();
    let sample_hz = audio_info.sample_rate() as usize;

    // Set our window for fourier transforms to 0.2 seconds. This might cause problems with the low
    // end couple of octaves, where the notes don't differ by more than a few Hz.
    let window_width = sample_hz / 5;

    // 'stride' is the number of samples that we move our window forward each frame.
    let stride = window_width / 4;

    let bin_width = sample_hz as f32 / (window_width as f32);

    let frames_range = 0..((samples.len() - window_width) / stride);

    // frames is a Vec<Vec<f32>>, where the first index corresponds to the frame number and the
    // second index corresponds to the bin number from the frequency spectrum.
    let frames = frames_range
        .map(|frame_num| {
            let window_start = frame_num * stride;
            let window_end = window_start + window_width;

            do_fourier_transform(&samples[window_start..window_end]).collect_vec()
        })
        .collect_vec();

    assert!(!frames.is_empty());
    let bin_count = frames[0].len();

    let threshold = 50.0_f32;

    // Process the data in 'frames' into a time series of Vec<bool> which indicates where
    // each note was played. Drop the first/last frames/bins to make the inner code a bit nicer.
    let note_frames = (1..frames.len() - 1)
        .map(|frame| {
            (1..bin_count - 1)
                .map(|bin| {
                    if frames[frame - 1][bin] > frames[frame][bin]
                        || frames[frame + 1][bin] > frames[frame][bin]
                    {
                        return (bin, false);
                    }

                    if frames[frame][bin - 1] > frames[frame][bin]
                        || frames[frame][bin + 1] > frames[frame][bin]
                    {
                        return (bin, false);
                    }

                    (bin, frames[frame][bin] > threshold)
                })
                .filter_map(|(bin, spike)| {
                    if spike {
                        closest_note(bin as f32 * bin_width)
                    } else {
                        None
                    }
                })
                .collect_vec()
        })
        .collect_vec();

    for (index, note_frame) in note_frames.iter().enumerate() {
        let frame_start_time = index as f32 * stride as f32 / sample_hz as f32;
        print!("Frame {} ({}):", index, frame_start_time);

        for note in note_frame {
            print!(" {}", get_note_name(*note));
        }

        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_select_notes<'a>(
        bin_width: f32,
        frequency_spectrum_iter: impl Iterator<Item = f32> + 'a,
    ) -> impl Iterator<Item = f32> + 'a {
        // TODO bin_width is generally going to be too low to distinguish between low-end notes. Give
        // the user a warning here that indicates the range of notes that might be incorrect due to
        // fourier transform inaccuracies in the low range.

        let spectrum = frequency_spectrum_iter.collect_vec();

        // Notes frequently lie "between" bins, so take the weighted average of the
        // norms of the adjacent bins when computing a note's magnitude.
        let weighted_index = move |bin: f32| -> f32 {
            let first = spectrum[bin as usize];
            let second = spectrum[bin as usize + 1];

            let interp = bin - bin.floor();

            first.mul_add(1.0 - interp, second * interp)
        };

        // Now that we have the frequency spectrum, select the bins that correspond
        // to notes.
        NOTES_RANGE.map(move |note| weighted_index(get_note_freq(note) / bin_width))
    }

    // Generate a tone using the given frequencies, at the given sample rate, for the given number
    // of samples.
    fn make_tone_iter(
        frequencies: &[f32],
        sample_rate: usize,
        num_samples: usize,
    ) -> impl Iterator<Item = f32> + '_ {
        use std::f32::consts::PI;

        (0..num_samples).into_iter().map(move |x| {
            let theta = 2.0 * PI * x as f32 / sample_rate as f32;

            frequencies.iter().map(|f| (f * theta).sin()).sum::<f32>() / (frequencies.len() as f32)
        })
    }

    fn do_tone_test(sample_hz: usize, sample_count: usize, notes: &Vec<Note>) {
        let frequencies = notes.iter().map(|note| get_note_freq(*note)).collect_vec();

        let samples = make_tone_iter(&frequencies, sample_hz, sample_count).collect_vec();

        let threshold = 50.0_f32;

        let bin_width = sample_hz as f32 / (samples.len() as f32);

        assert_eq!(
            NOTES_RANGE
                .zip(naive_select_notes(
                    bin_width,
                    do_fourier_transform(&samples[..])
                ))
                .filter(|(_note, mag)| *mag > threshold)
                .map(|(x, _y)| x)
                .collect_vec(),
            *notes
        );
    }

    #[test]
    fn test_harmonic_scale() {
        // Harmonic scale starting from A4
        do_tone_test(22050, 7350, &(0..12).collect_vec());
    }

    #[test]
    fn test_major_scale() {
        // Do a major scale starting from middle C
        do_tone_test(22050, 7350, &vec![-9, -7, -5, -4, -2, 0, 2, 3]);
    }

    #[test]
    fn sanity_num_notes() {
        assert_eq!(NOTES_RANGE.count(), 88)
    }
}
