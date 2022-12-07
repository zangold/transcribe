use itertools::Itertools;
use std::error::Error;

use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;

//use rustfft::{num_complex::Complex, num_traits::Zero};

type Note = i32;

fn get_note_freq(note: Note) -> f32 {
    440.0_f32 * 2.0_f32.powf(note as f32 / 12.0_f32)
}

fn get_note_name(note: Note) -> String {
    assert!(-48 <= note && note <= 39);

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

    //println!("audio: sample rate {} Hz", audio_info.sample_rate());
    //println!("       channels: {}", audio_info.channels());
    //println!("       format: {:?}", audio_info.format());

    Ok((audio_info, samples))
}

/// Does some preliminary processing on the SampleIterator to make the upcoming
/// work easier:
/// - unwraps the Results for each sample
/// - merges two channels into one, if two channels are present
fn merge_audio(
    info: &creak::AudioInfo,
    samples: creak::SampleIterator,
) -> impl Iterator<Item = f32> + '_ {
    samples
        .into_iter()
        .map(|x| x.unwrap())
        .batching(|it| match info.channels() {
            1 => it.next(),
            2 => match it.next() {
                None => None,
                Some(l) => match it.next() {
                    None => None,
                    Some(r) => Some((l + r) / 2.0),
                },
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

        println!("");
    }
}

/// Convert the given samples into the frequency spectrum and print the results
fn do_fourier_transform(samples: &[f32], sample_frequency: usize) -> impl Iterator<Item = (Note, f32)> + 'static {
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

    // Notes frequently lie "between" bins, so take the weighted average of the
    // norms of the adjacent bins when computing a note's magnitude.
    let weighted_index = |bin: f32| -> f32 {
        let first = spectrum[bin as usize].norm();
        let second = spectrum[bin as usize + 1].norm();

        let interp = bin - bin.floor();

        first * (1.0 - interp) + second * interp
    };

    // Now that we have the frequency spectrum, select the bins that correspond
    // to notes.
    let note_magnitudes = (-48..39)
        .map(|note| weighted_index(get_note_freq(note) / bin_width))
        .collect_vec();

    // Have to define my own function for this. reduce() expects a function that takes references
    // to f32, but f32::max takes just f32, and rust can't fill in the gaps.
    //let max_f32 = |x, y| if x > y { x } else { y };

    //let max_magnitude = *note_magnitudes.iter().reduce(max_f32).unwrap();

    (-48..39).zip(note_magnitudes.into_iter())
}

fn make_tone_iter<'a>(
    frequencies: &'a [f32],
    sample_rate: usize,
    num_samples: usize,
) -> impl Iterator<Item = f32> + 'a {
    use std::f32::consts::PI;

    (0..num_samples).into_iter().map(move |x| {
        let theta = 2.0 * PI * x as f32 / sample_rate as f32;

        frequencies.iter().map(|f| (f * theta).sin()).sum::<f32>() / (frequencies.len() as f32)
    })
}

fn main() {
    let (audio_info, samples) = load("sample.mp3").unwrap();

    let _samples = merge_audio(&audio_info, samples);

    // For testing: do a harmonic scale starting from "A above middle C"
    let sample_hz = 22050;
    let sample_count = 7350;
    let frequencies = (0..12)
        .into_iter()
        .map(|note| get_note_freq(note))
        .collect_vec();
    let samples = make_tone_iter(&frequencies, sample_hz, sample_count).collect_vec();

    let notes = do_fourier_transform(&samples[..], sample_hz)
        .into_iter()
        .filter(|(_note, mag)| *mag > 50.0_f32)
        .map(|(x, _y)| x);

    for note in notes {
        println!("Sample contained note {}", get_note_name(note));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn do_tone_test(sample_hz: usize, sample_count: usize, notes: &Vec<Note>) {
        let frequencies = notes.iter().map(|note| get_note_freq(*note)).collect_vec();

        let samples = make_tone_iter(&frequencies, sample_hz, sample_count).collect_vec();

        let threshold = 50.0_f32;

        assert_eq!(
            do_fourier_transform(&samples[..], sample_hz)
                .into_iter()
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
}
