'''
Definition of Shinra Attribute Extraction 2022 dataset
'''

import json
import logging
import os

import datasets
from datasets import (
    DatasetInfo,
    DownloadManager,
    GeneratorBasedBuilder,
    SplitGenerator,
)

#from logzero import logger
logger = logging.getLogger(__name__)

class ShinraAttributeExtraction2022Builder(GeneratorBasedBuilder):
    '''
    Dataset Class for Shinra Attribute Extraction 2022
    '''
    _URL = "https://storage.googleapis.com/shinra_data/data/"
    _URLS = {
        "train": _URL + "attribute_extraction-20221116.zip",
    }

    def _info(self):
        return DatasetInfo(
            features=datasets.Features(
                {
                    "page_id": datasets.Value("string"),
                    "category_name": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "attribute": datasets.Value("string"),
                    "ENE": datasets.Value("string"),
                    "html": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "offsets": datasets.Sequence({
                        "html_offset": {
                            "start": {
                                "line_id": datasets.Value("int32"),
                                "offset": datasets.Value("int32"),
                            },
                            "end": {
                                "line_id": datasets.Value("int32"),
                                "offset": datasets.Value("int32"),
                            },
                            "text": datasets.Value("string"),
                        },
                        "text_offset": {
                            "start": {
                                "line_id": datasets.Value("int32"),
                                "offset": datasets.Value("int32"),
                            },
                            "end": {
                                "line_id": datasets.Value("int32"),
                                "offset": datasets.Value("int32"),
                            },
                            "text": datasets.Value("string"),
                        },
                    })
                }
            )
        )

    def _split_generators(self, dl_manager: DownloadManager):
        urls_to_download = self._URLS
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "filepath": downloaded_files["train"],
                },
            ),
        ]
    
    def _find_dist_directory(self, dirpath):
        #logger.debug('_find_dist_directory dirpath: %s', dirpath)
        if all(
            os.path.isdir(os.path.join(dirpath, dirname))
            for dirname in ['annotation', 'html', 'plain']
        ):
            return dirpath

        for entry in os.listdir(dirpath):
            if os.path.isdir(os.path.join(dirpath, entry)):
                dist_directory = self._find_dist_directory(os.path.join(dirpath, entry))
                if dist_directory:
                    return dist_directory
        return None
    
    def _generate_examples_for_train(self, dirpath):
        if not os.path.isdir(dirpath):
            raise ValueError(f'"{dirpath}" is not a directory')
        
        dist_dir = self._find_dist_directory(dirpath)
        if not dist_dir:
            raise ValueError(f'Could not find dist directory in "{dirpath}"')
        
        annotation_dir = os.path.join(dist_dir, 'annotation')
        html_dir = os.path.join(dist_dir, 'html')
        plain_dir = os.path.join(dist_dir, 'plain')

        for entry in os.listdir(annotation_dir):
            fullpath = os.path.join(annotation_dir, entry)
            if os.path.isfile(fullpath):
                if entry.endswith('_dist.jsonl'):
                    category = entry.split('_dist.jsonl')[0]
                    map_page_id_to_records: dict[str, list] = {}
                    with open(fullpath, 'r', encoding='utf-8') as f:
                        for line in f:
                            data: dict = json.loads(line)
                            page_id = data['page_id']
                            if page_id not in map_page_id_to_records:
                                map_page_id_to_records[page_id] = []
                            map_page_id_to_records[page_id].append(data)
                    for page_id, records in map_page_id_to_records.items():
                        first = records[0]
                        offsets = []
                        html_path = os.path.join(html_dir, category, page_id + '.html')
                        text_path = os.path.join(plain_dir, category, page_id + '.txt')
                        with open(html_path, 'r', encoding='utf-8') as f:
                            html = f.read()
                        with open(text_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        for record in records:
                            offsets.append({
                                'html_offset': record['html_offset'],
                                'text_offset': record['text_offset'],
                            })
                        yield page_id, {
                            'page_id': page_id,
                            'category_name': category,
                            'title': first['title'],
                            'attribute': first['attribute'],
                            'ENE': first['ENE'],
                            'offsets': offsets,
                            'html': html,
                            'text': text,
                        }
                    #break

    def _generate_examples(self, **kwargs):
        split = kwargs["split"]
        filepath = kwargs["filepath"]
        #logger.debug('filepath: %s', filepath)
        #logger.debug('split: %s', split)

        if split == "train":
            for id_, example in self._generate_examples_for_train(filepath):
                yield id_, example
