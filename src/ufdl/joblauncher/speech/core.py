import traceback

from ufdl.joblauncher.core.executors import AbstractJobExecutor

from ufdl.pythonclient.functional.speech.dataset import set_transcription_for_file


def store_transcription(
        executor: AbstractJobExecutor,
        dataset_pk: int,
        file_name: str,
        transcription: str
):
    """
    Stores the annotations in the backend.

    :param executor: the executor class this is done for
    :param dataset_pk: the PK of the dataset these scores are calculated for
    :param file_name: the name of the file the transcription is for
    :param transcription: the file's transcription
    """
    # set transcription for file
    try:
        set_transcription_for_file(executor.context, dataset_pk, file_name, transcription)
    except:
        executor.log_msg(
            f"Failed to add annotations generated by job {executor.job_pk} to dataset {dataset_pk}!\n"
            f"{traceback.format_exc()}"
        )
