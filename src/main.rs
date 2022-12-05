use itertools::Itertools;
use std::error::Error;

use realfft::RealFftPlanner;

//use rustfft::{num_complex::Complex, num_traits::Zero};

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

/// Convert the given samples into the frequency spectrum and print the results
fn do_fourier_transform(samples: &[f32], sample_frequency: usize) {
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

    //println!("Fourier transform output: {:?}", spectrum);

    let bin_width = sample_frequency as f32 / (samples.len() as f32);

    println!("FFT bin width: {bin_width}");

    for (index, bin) in spectrum.iter().enumerate() {
        // figure out the magnitude of the complex number, then print one # per 0.1?
        let magnitude = bin.norm();
        print!("bin {index}: ");

        for _ in 0..magnitude as usize {
            print!("#");
        }

        println!("");
    }
}

fn make_tone_iter<'a> (
    frequencies: &'a [f32],
    sample_rate: usize,
    num_samples: usize,
) -> impl Iterator<Item = f32> + 'a {
    use std::f32::consts::PI;

    (0..num_samples)
        .into_iter()
        .map(move |x| {
            let theta = 2.0 * PI * x as f32 / sample_rate as f32;

            frequencies.iter().map(|f| (f * theta).sin()).sum::<f32>() / (frequencies.len() as f32)
        })
}

fn main() {
    let (audio_info, samples) = load("sample.mp3").unwrap();

    let _samples = merge_audio(&audio_info, samples);

    //for sample in samples {
    //    println!("{sample}");
    //}

    // Test fourier transform. Generate a 440Hz tone and see what comes out of
    // the fft algorithm.
    let sample_hz = 22050;
    let sample_count = 7350;
    let frequencies = (0..12).into_iter().map(|x| 256.0 * 2.0_f32.powf(x as f32 / 12.0)).collect_vec();
    let samples = make_tone_iter(&frequencies, sample_hz, sample_count).collect_vec();

    do_fourier_transform(&samples[..], sample_hz);
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
}
