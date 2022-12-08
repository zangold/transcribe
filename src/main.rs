use collect_slice::CollectSlice;
use itertools::Itertools;
use std::cmp::min;
use std::error::Error;

use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;

type Note = i32;

// A0 to C8
const NOTES_RANGE: std::ops::Range<i32> = -48..40;

const NUM_NOTES: usize = (NOTES_RANGE.end - NOTES_RANGE.start) as usize;

/// Computes the frequency of the given note, where 'note' is an integer representing the number of
/// semitones above (+ve) or below (-ve) A4. Assumes A4 is tuned to 440Hz and notes are distributed
/// according to even temperament (semitones differ by the 12th root of 2).
fn get_note_freq(note: Note) -> f32 {
    440.0_f32 * 2.0_f32.powf(note as f32 / 12.0_f32)
}

/// Convert a note value into an index into an array of notes; i.e., map from NOTES_RANGE to (0, 88)
fn note_index(note: Note) -> usize {
    (note - NOTES_RANGE.start) as usize
}

#[allow(dead_code)]
fn get_note_name(note: Note) -> String {
    assert!(NOTES_RANGE.contains(&note));

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

fn load(filename: &str) -> Result<(creak::AudioInfo, creak::SampleIterator), Box<dyn Error>> {
    let decoder = creak::Decoder::open(filename)?;

    let audio_info = decoder.info();
    let samples = decoder.into_samples()?;

    println!("audio: sample rate {} Hz", audio_info.sample_rate());
    println!("       channels: {}", audio_info.channels());
    println!("       format: {:?}", audio_info.format());

    Ok((audio_info, samples))
}

/// Does some preliminary processing on the SampleIterator to make the upcoming
/// work easier:
/// - unwraps the Results for each sample
/// - merges two channels into one, if two channels are present
fn merge_audio(
    info: &creak::AudioInfo,
    samples: creak::SampleIterator,
) -> impl Iterator<Item = f32> + 'static {
    let channels = info.channels();
    samples
        .into_iter()
        .map(|x| x.unwrap())
        .batching(move |it| match channels {
            1 => it.next(),
            2 => match it.next() {
                None => None,
                Some(l) => it.next().map(|_r| l),
            },
            x => panic!("Invalid number of channels: {x}"),
        })
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
fn do_fourier_transform(
    samples: &[f32],
    sample_frequency: usize,
) -> impl Iterator<Item = f32> + 'static {
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

    let bin_width = sample_frequency as f32 / (samples.len() as f32);

    // TODO bin_width is generally going to be too low to distinguish between low-end notes. Give
    // the user a warning here that indicates the range of notes that might be incorrect due to
    // fourier transform inaccuracies in the low range.

    // Notes frequently lie "between" bins, so take the weighted average of the
    // norms of the adjacent bins when computing a note's magnitude.
    let weighted_index = move |bin: f32| -> f32 {
        let first = spectrum[bin as usize].norm();
        let second = spectrum[bin as usize + 1].norm();

        let interp = bin - bin.floor();

        first * (1.0 - interp) + second * interp
    };

    // Now that we have the frequency spectrum, select the bins that correspond
    // to notes.
    NOTES_RANGE.map(move |note| weighted_index(get_note_freq(note) / bin_width))
}

fn main() {
    let (audio_info, samples) = load("canon_d_kassia.mp3").unwrap();

    let samples = merge_audio(&audio_info, samples).collect_vec();
    let sample_hz = audio_info.sample_rate() as usize;

    // Set our window for fourier transforms to 0.2 seconds. This might cause problems with the low
    // end couple of octaves, where the notes don't differ by more than a few Hz.
    let num_samples = sample_hz / 5;

    // 'stride' is the number of samples that we move our window forward each frame.
    let stride = num_samples / 4;

    let mut window_start = 0;
    let mut note_mag_time_series = Vec::<[f32; NUM_NOTES]>::new();
    let mut series_index = 0;

    println!(
        "Info: allocating {} bytes for (note/magnitude)-over-time data",
        std::mem::size_of::<[f32; NUM_NOTES]>() * (samples.len() / stride)
    );

    note_mag_time_series.resize(samples.len() / stride, [0.0_f32; NUM_NOTES]);

    while window_start + num_samples < samples.len() {
        do_fourier_transform(
            &samples[window_start..window_start + num_samples],
            sample_hz,
        )
        .collect_slice_checked(&mut note_mag_time_series[series_index][..]);

        window_start += stride;
        series_index += 1;
    }

    let mut note_time_series = Vec::<[bool; NUM_NOTES]>::new();
    note_time_series.resize(note_mag_time_series.len(), [false; NUM_NOTES]);

    for frame_index in 1..note_mag_time_series.len() - 1 {
        let prev_frame = &note_mag_time_series[frame_index - 1];
        let frame = &note_mag_time_series[frame_index];
        let next_frame = &note_mag_time_series[frame_index + 1];

        for note in NOTES_RANGE.start + 1..NOTES_RANGE.end - 1 {
            let note_index = note_index(note);

            // threshold check: if this note was super quiet, then it probably wasn't played.
            if frame[note_index] < 5.0_f32 {
                continue;
            }

            // neighbour check: if the adjacent notes are quieter than this one, this probably
            // was played.
            if !(frame[note_index - 1] < frame[note_index]
                && frame[note_index] > frame[note_index + 1])
            {
                continue;
            }

            // temporal check: if this note was quieter in the last frame, and is quieter in
            // the next frame, then it was likely to just have been played.
            if !(prev_frame[note_index] < frame[note_index]
                && frame[note_index] > next_frame[note_index])
            {
                continue;
            }

            note_time_series[frame_index][note_index] = true;
        }
    }

    // Output the time series as CSV.
    print!("time,");
    for i in 0..min(100, note_mag_time_series.len()) {
        let time = (i * stride) as f32 / (sample_hz as f32);

        print!("{time},");
    }
    println!();

    for note in NOTES_RANGE {
        let name = get_note_name(note);
        print!("{},", name);

        for i in 0..min(100, note_mag_time_series.len()) {
            if note_time_series[i][note_index(note)] {
                print!("{}({}),", name, note_mag_time_series[i][note_index(note)]);
            } else {
                print!(",");
            }
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        assert_eq!(
            NOTES_RANGE
                .zip(do_fourier_transform(&samples[..], sample_hz))
                .filter(|(_note, mag)| *mag > threshold)
                .map(|(x, _y)| x)
                .collect_vec(),
            *notes
        );
    }

    #[test]
    /// Test to make sure that merge_audio yields the correct number of samples.
    fn test_merge_audio() {
        let (audio_info, samples) = load("sample.mp3").unwrap();
        let num_samples = samples.into_iter().count();

        assert_eq!(audio_info.channels(), 2);

        let (audio_info, samples) = load("sample.mp3").unwrap();
        let merged_samples = merge_audio(&audio_info, samples);

        let num_merged_samples = merged_samples.count();

        assert_eq!(num_samples / 2, num_merged_samples);
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
        assert_eq!(NUM_NOTES, 88);
        assert_eq!(NOTES_RANGE.count(), 88)
    }
}
