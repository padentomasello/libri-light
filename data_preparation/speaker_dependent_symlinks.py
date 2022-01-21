# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path, PurePath, PurePosixPath
import metadata_completion.utilities as ut
import torchaudio
from collections import defaultdict
import progressbar
import os
import json
import plot

def create_symlinks(speaker_splits, base, out_dir):
    for speaker, splits in speaker_splits.items():
        output = Path(out_dir)
        for k in ("train", "val", "test", "speaker_independent"):
            for x in splits[k]:
                dst = output / k
                dst = dst /  (x.replace(base, ""))
                dst.parent.mkdir(parents=True, exist_ok=-True)
                meta_src = PurePath(x).with_suffix('.json')
                meta_dst = dst.with_suffix('.json')
                dst.symlink_to(x)
                meta_dst.symlink_to(meta_src)


def create_speaker_dependent_splits(list_metadata, audio_extension, librispeech_splits,
        val_hours=1, test_hours=1):

    speakerTalk = {}
    nData = len(list_metadata)

    bar = progressbar.ProgressBar(maxval=nData)
    bar.start()

    speaker_splits = defaultdict(lambda :{
        "train": [], 
        "val": [], 
        "test":[], 
        "speaker_independent": [],
        "librispeech_train_speakers": []
        })

    for index, pathMetadata in enumerate(list_metadata):
        bar.update(index)
        with open(pathMetadata, 'rb') as file:
            locMetadata = json.load(file)

        speaker_name = locMetadata['speaker']

        path_audio_data = os.path.splitext(pathMetadata)[0] + audio_extension

        info = torchaudio.info(path_audio_data)
        totAudio = info.num_frames / (info.sample_rate * 3600.)

        if speaker_name is None:
            speaker_name = 'null'

        if speaker_name not in speakerTalk:
            speakerTalk[speaker_name] = 0

        speakerTalk[speaker_name] += totAudio

        time_so_far = speakerTalk[speaker_name]
        splits = speaker_splits[speaker_name]

        assert(speaker_name not in librispeech_splits["dev"])
        assert(speaker_name not in librispeech_splits["test"])

        # import pdb; pdb.set_trace()
        if (int(speaker_name) in librispeech_splits["train"]):
            splits["librispeech_train_speakers"].append(path_audio_data)
            # print("Found speaker in librispeech train")
        else:
            if time_so_far < test_hours:
                splits["test"].append(path_audio_data)
            elif time_so_far < (val_hours + test_hours):
                splits["val"].append(path_audio_data)
            else:
                splits["train"].append(path_audio_data)

    for speaker_id, splits in speaker_splits.items():
        if len(splits["train"]) == 0 and speaker_id not in librispeech_splits["train"]:
            splits["speaker_independent"].extend(splits["val"])
            splits["speaker_independent"].extend(splits["test"])
            splits["val"] = []
            splits["test"] = []
            # print("Not enough val and test data!")

    for speaker_id, splits in speaker_splits.items():
        if(len(splits["librispeech_train_speakers"]) > 0):
            assert(len(splits["speaker_independent"]) == 0)
            assert(len(splits["train"]) == 0)
            assert(len(splits["val"]) == 0)
            assert(len(splits["test"]) == 0)
        elif(len(splits["speaker_independent"]) > 0):
            assert(len(splits["train"]) == 0)
            assert(len(splits["val"]) == 0)
            assert(len(splits["test"]) == 0)
        else:
            print(f'Speaker dependent split for speaker: ${speaker_id} with total hours: ${speakerTalk[speaker_id]}')

    bar.finish()
    return speaker_splits

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build the statistics on LibriBig")
    parser.add_argument('path_data', type=str,
                        help="Path to the directory containing the data")
    parser.add_argument('out_dir', type=str,
                        help="Path to the output directory")
    parser.add_argument('--ignore_cache', action='store_true')
    args = parser.parse_args()

    # Build the output directory
    args.out_dir = Path(args.out_dir)
    Path.mkdir(args.out_dir, exist_ok=True)

    # Build the cache directory
    path_cache = args.out_dir / ".cache"
    Path.mkdir(path_cache, exist_ok=True)

    # Get the list of all metadata
    print("Gathering the list of metadata")
    path_cache_metadata = path_cache / "metadata.pkl"
    list_metadata = ut.load_cache(path_cache_metadata,
                                  ut.get_all_metadata,
                                  args=(args.path_data, ".json"),
                                  ignore_cache=args.ignore_cache)
    print(f"{len(list_metadata)} files found")


    # Get the speaker statistics
    print("Building the speaker statistics")
    path_speaker_cache = path_cache / "speaker_stats.json"
    with open("speakers.json") as f:
        librispeech_splits = json.load(f)
    speaker_splits = create_speaker_dependent_splits(list_metadata, ".flac", librispeech_splits)
    # create_symlinks(speaker_splits, args.path_data, args.out_dir)

    print("done.")
