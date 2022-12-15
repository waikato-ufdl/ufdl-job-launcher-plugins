import shlex
import sys
from glob import glob
import os
import traceback
from typing import Tuple, Union

from ufdl.jobcontracts.standard import Train, Predict

from ufdl.joblauncher.core.executors import AbstractJobExecutor
from ufdl.joblauncher.core.executors._util import download_dataset
from ufdl.joblauncher.core.executors.descriptors import Parameter, ExtraOutput

from ufdl.jobtypes.base import Boolean, String
from ufdl.jobtypes.standard.container import Array
from ufdl.jobtypes.standard.server import Domain, Framework
from ufdl.jobtypes.standard.util import BLOB

from ufdl.pythonclient.functional.spectrum_classification.dataset import add_categories

from ...utils import write_to_file


DOMAIN_TYPE = Domain("Spectrum Classification")
FRAMEWORK_TYPE = Framework("weka", "classifier")
SPECTRUM_CLASSIFICATION_WEKA_CONTRACT_TYPES = {'DomainType': DOMAIN_TYPE, 'FrameworkType': FRAMEWORK_TYPE}


class SpectrumClassificationTrain_Weka(AbstractJobExecutor[Train]):
    """
    Trial dummy learner for the dog-breeds dataset.
    """
    _cls_contract = Train(SPECTRUM_CLASSIFICATION_WEKA_CONTRACT_TYPES)

    classname: str = Parameter(String())

    options: Union[str, Tuple[str, ...]] = Parameter(
        String(),
        Array(String())
    )

    def _pre_run(self):
        """
        Hook method before the actual job is run.

        :return: whether successful
        :rtype: bool
        """
        if not super()._pre_run():
            return False

        # create directories
        self._mkdir(self.job_dir + "/data")

        # download dataset
        download_dataset(
            self.context,
            self[self.contract.dataset].pk,
            self.template['domain'],
            self.job_dir + "/data",
            ["to-subdir-sc", "-o", "."]
        )

        self.progress(0.1, comment="Downloaded dataset")

        return True

    def _do_run(self):
        """
        Executes the actual job. Only gets run if pre-run was successful.
        """
        # build model
        res = self._execute(
            [
                sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "bare.py"),
                "train",
                self.job_dir + "/data",
                self.job_dir,
                self.classname,
                (
                    self.options if isinstance(self.options, str)
                    else ' '.join(shlex.quote(option) for option in self.options)
                )
            ]
        )

        if res is not None:
            res.check_returncode()

        self.progress(0.9, comment="Analysed dataset")

    def _post_run(self, pre_run_success, do_run_success, error):
        """
        Hook method after the actual job has been run. Will always be executed.

        :param pre_run_success: whether the pre_run code was successfully run
        :type pre_run_success: bool
        :param do_run_success: whether the do_run code was successfully run (only gets run if pre-run was successful)
        :type do_run_success: bool
        :param error: any error that may have occurred, None if none occurred
        :type error: str
        """
        # zip+upload model
        if do_run_success:
            self._compress_and_upload(
                self.contract.model,
                [
                    self.job_dir + "/model",
                    self.job_dir + "/labels"
                ],
                self.job_dir + "/model.zip"
            )

        super()._post_run(pre_run_success, do_run_success, error)


class SpectrumClassificationPredict_Weka(AbstractJobExecutor[Predict]):
    """
    Trial dummy predictor for the dog-breeds dataset.
    """
    _cls_contract = Predict(SPECTRUM_CLASSIFICATION_WEKA_CONTRACT_TYPES)

    clear_dataset: bool = Parameter(Boolean())

    store_predictions: bool = Parameter(Boolean())

    predictions = ExtraOutput(BLOB("txt"))

    def _pre_run(self):
        """
        Hook method before the actual job is run.

        :return: whether successful
        :rtype: bool
        """
        if not super()._pre_run():
            return False

        # create directories
        self._mkdir(self.job_dir + "/prediction")
        self._mkdir(self.job_dir + "/prediction/in")
        self._mkdir(self.job_dir + "/prediction/out")

        # download dataset
        download_dataset(
            self.context,
            self[self.contract.dataset].pk,
            self.template['domain'],
            self.job_dir + "/prediction/in",
            ["to-spectra-sc"],
            self.clear_dataset
        )

        self.progress(0.05, comment="Downloaded dataset")

        # download model
        model = self.job_dir + "/model.zip"
        with open(model, "wb") as zip_file:
            write_to_file(zip_file, self[self.contract.model])

        self.progress(0.1, comment="Downloaded model")

        # decompress model
        msg = self._decompress(model, self.job_dir)
        if msg is not None:
            raise Exception(f"Failed to extract model!\n{msg}")

        self.progress(0.15, comment="Decompressed model")

        return True

    def _do_run(self):
        """
        Executes the actual job. Only gets run if pre-run was successful.
        """
        res = self._execute(
            [
                sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "bare.py"),
                "predict",
                self.job_dir,
                self.job_dir + "/prediction/in",
                self.job_dir + "/prediction/out"
            ]
        )

        if res is not None:
            res.check_returncode()

    def _post_run(self, pre_run_success, do_run_success, error):
        """
        Hook method after the actual job has been run. Will always be executed.

        :param pre_run_success: whether the pre_run code was successfully run
        :type pre_run_success: bool
        :param do_run_success: whether the do_run code was successfully run (only gets run if pre-run was successful)
        :type do_run_success: bool
        :param error: any error that may have occurred, None if none occurred
        :type error: str
        """
        dataset_pk: int = self[self.contract.dataset].pk

        # zip+upload predictions
        if do_run_success:
            self._compress_and_upload(
                self.predictions,
                glob(self.job_dir + "/prediction/out/*.txt"),
                self.job_dir + "/predictions.zip")

        # post-process predictions
        if do_run_success and self.store_predictions:
            try:
                for f in glob(self.job_dir + "/prediction/in/**/*", recursive=True):
                    if not f.endswith(".spec"):
                        continue
                    spec_name = os.path.basename(f)

                    # load TXT file and determine label
                    txt_file = self.job_dir + f"/prediction/out/{os.path.splitext(spec_name)[0]}.txt"
                    with open(txt_file, "r") as file:
                        label = file.read()

                    # set category for file
                    if label != "":
                        try:
                            add_categories(self.context, dataset_pk, [spec_name], [label])
                        except:
                            self.log_msg("Failed to add labels generated by job %d to dataset %d!\n%s" % (self.job_pk, dataset_pk, traceback.format_exc()))

            except:
                self.log_msg("Failed to post-process predictions generated by job %d for dataset %d!\n%s" % (self.job_pk, dataset_pk, traceback.format_exc()))

        self.progress(1.0, comment="Uploaded predictions to server")

        super()._post_run(pre_run_success, do_run_success, error)
