set -e -x
spectrogram_generators=(tts_en_fastpitch tts_en_tacotron2 tts_en_glowtts)
vocoders=(tts_waveglow_88m tts_squeezewave tts_uniglow tts_melgan tts_hifigan)

output_dir=~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/test_tts_2_stages
audio_dir="${output_dir}/audio"

wer_file="${output_dir}/wer.txt"
rm -r "${audio_dir}"/* "${output_dir}"/asr_preds__* "${output_dir}/asr_references.txt" "${wer_file}"
rm -r

for spectrogram_generator in "${spectrogram_generators[@]}"; do
  for vocoder in "${vocoders[@]}"; do
    python test_tts_infer.py \
      --tts_model_spec "${spectrogram_generator}" \
      --tts_model_vocoder "${vocoder}" \
      --input ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/iwslt_en_text.txt \
      --asr_preds "${output_dir}/asr_preds__${spectrogram_generator}__${vocoder}.txt" \
      --asr_references "${output_dir}/asr_references.txt" \
      --audio_preds "${audio_dir}/${spectrogram_generator}__${vocoder}" \
      --wer_file "${wer_file}"
  done
done
set +e +x