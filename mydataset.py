from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """\
Large Movie Review Dataset.
This is a dataset for binary sentiment classification containing substantially \
more data than previous benchmark datasets. We provide a set of 25,000 highly \
polar movie reviews for training, and 25,000 for testing. There is additional \
unlabeled data for use as well.\
"""

_CITATION = """\
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
"""

_DOWNLOAD_URL = "https://github.com/Anaghas27/Dataset/blob/master/Dataset.zip"


class MyDatasetConfig(tfds.core.BuilderConfig):
  """BuilderConfig for IMDBReviews."""

  @tfds.core.disallow_positional_args
  def __init__(self, text_encoder_config=None, **kwargs):
    """BuilderConfig for IMDBReviews.
    Args:
      text_encoder_config: `tfds.features.text.TextEncoderConfig`, configuration
        for the `tfds.features.text.TextEncoder` used for the IMDB `"text"`
        feature.
      **kwargs: keyword arguments forwarded to super.
    """
    super(MyDatasetConfig, self).__init__(
        version=tfds.core.Version(
            "1.0.0",
            "New split API (https://tensorflow.org/datasets/splits)"),
        **kwargs)
    self.text_encoder_config = (
        text_encoder_config or tfds.features.text.TextEncoderConfig())


class MyDataset(tfds.core.GeneratorBasedBuilder):
  """IMDB movie reviews dataset."""
  BUILDER_CONFIGS = [
      MyDatasetConfig(
          name="plain_text",
          description="Plain text",
      ),
      MyDatasetConfig(
          name="bytes",
          description=("Uses byte-level text encoding with "
                       "`tfds.features.text.ByteTextEncoder`"),
          text_encoder_config=tfds.features.text.TextEncoderConfig(
              encoder=tfds.features.text.ByteTextEncoder()),
      ),
      MyDatasetConfig(
          name="subwords8k",
          description=("Uses `tfds.features.text.SubwordTextEncoder` with 8k "
                       "vocab size"),
          text_encoder_config=tfds.features.text.TextEncoderConfig(
              encoder_cls=tfds.features.text.SubwordTextEncoder,
              vocab_size=2**13),
      ),
      MyDatasetConfig(
          name="subwords32k",
          description=("Uses `tfds.features.text.SubwordTextEncoder` with "
                       "32k vocab size"),
          text_encoder_config=tfds.features.text.TextEncoderConfig(
              encoder_cls=tfds.features.text.SubwordTextEncoder,
              vocab_size=2**15),
      ),
  ]

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "text": tfds.features.Text(
                encoder_config=self.builder_config.text_encoder_config),
            "label": tfds.features.ClassLabel(names=["negative", "positive"]),
        }),
        supervised_keys=("text", "label"),
        citation=_CITATION,
    )

  def _vocab_text_gen(self, archive):
    for _, ex in self._generate_examples(
        archive, os.path.join("MyData", "train")):
      yield ex["text"]

  def _split_generators(self, dl_manager):
    arch_path = dl_manager.download(_DOWNLOAD_URL)
    archive = lambda: dl_manager.iter_archive(arch_path)

    # Generate vocabulary from training data if SubwordTextEncoder configured
    self.info.features["text"].maybe_build_from_corpus(
        self._vocab_text_gen(archive()))

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={"archive": archive(),
                        "directory": os.path.join("MyData", "train")}),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={"archive": archive(),
                        "directory": os.path.join("MyData", "test")}),
    ]

  def _generate_examples(self, archive, directory, labeled=True):
    """Generate IMDB examples."""
    # For labeled examples, extract the label from the path.
    reg_path = "(?P<label>negative|positive)" if labeled else "unsup"
    reg = re.compile(
        os.path.join("^%s" % directory, reg_path, "").replace("\\", "\\\\"))
    for path, imdb_f in archive:
      res = reg.match(path)
      if not res:
        continue
      text = imdb_f.read().strip()
      label = res.groupdict()["label"] if labeled else -1
      yield path, {
          "text": text,
          "label": label,
      }
