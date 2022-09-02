import nemo.collections.asr as nemo_asr
from corrector import SpellChecker

golos_model = 'QuartzNet15x5_golos.nemo'
restored_model = nemo_asr.models.EncDecCTCModel.restore_from('content/QuartzNet15x5_golos.nemo')
audio_files = ['content/1.wav']
checker = SpellChecker('dict.txt')
for transcription in restored_model.transcribe(audio_files):
    raw_transcription = transcription
    split_words = raw_transcription.split(' ')
    final_transcription = ''
    for word in split_words:
        final_transcription += checker.check(word) + ' '
    print(raw_transcription)
    print(final_transcription)