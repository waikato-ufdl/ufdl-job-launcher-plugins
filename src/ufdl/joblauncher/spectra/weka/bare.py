import math
import shlex
from glob import glob
import os
from types import TracebackType
from typing import ContextManager, Dict, Iterable, List, Optional, Set, Tuple, Type

from wai.common.file.spec import Spectrum, reader

from weka.core import jvm
from weka.classifiers import Classifier
from weka.core.dataset import Attribute, Instance, Instances
from weka.filters import Filter


class JVMContextManager(ContextManager[None]):
    """
    Context manager which automatically handles starting/stopping the JVM.
    """
    def __enter__(self):
        jvm.start()

    def __exit__(
            self,
            __exc_type: Optional[Type[BaseException]],
            __exc_value: Optional[BaseException],
            __traceback: Optional[TracebackType]
    ) -> Optional[bool]:
        jvm.stop()

        # The exception is unhandled, so re-raise
        return False


def create_instance(
        wave_numbers: List[float],
        spectrum: Spectrum,
        label_index: Optional[int]
) -> Instance:
    """
    Creates an instance for a spectrum.

    :param wave_numbers:
                The list of all wave-numbers that the spectrum could have.
    :param spectrum:
                The spectrum.
    :param label_index:
                The index of the spectrum's class label.
    :return:
                A representation of a spectrum as a WEKA instance.
    """
    # Create a lookup of amplitudes from wave-numbers for the spectrum
    amplitudes = {
        point.wave_number: point.amplitude
        for point in spectrum
    }

    # Create an instance matching the set of overall wave-number attributes,
    # setting the amplitude to missing value if the spectrum doesn't specify it
    return Instance.create_instance(
        [
            amplitudes.get(waveno, Instance.missing_value())
            for waveno in wave_numbers
        ] + [
            float(label_index) if label_index is not None
            else Instance.missing_value()
        ]
    )


def gather_wave_numbers(
        spectra: Iterable[Spectrum]
) -> List[float]:
    """
    Gathers all wave-numbers from a set of spectra into an ordered list.

    :param spectra:
                The spectra to gather wave-numbers from.
    :return:
                A list of wave-numbers in ascending order.
    """
    wave_numbers: List[float] = list({
        point.wave_number
        for spectrum in spectra
        for point in spectrum
    })
    wave_numbers.sort()
    return wave_numbers


def create_attributes(
        wave_numbers: List[float],
        labels: List[str]
) -> List[Attribute]:
    """
    Creates the attributes for a spectral dataset. This is a list of numeric
    attributes, one for each wave-number, followed by a final nominal attribute
    for the label.

    :param wave_numbers:
                The wave-numbers to create numeric attributes for.
    :param labels:
                The labels of the nominal attribute.
    :return:
                The list of attributes.
    """
    return [
        Attribute.create_numeric(str(wave_number))
        for wave_number in wave_numbers
    ] + [
        Attribute.create_nominal("label", labels)
    ]


def create_dataset(
        wave_numbers: List[float],
        labels: List[str],
        capacity: int
) -> Instances:
    """
    Creates a spectral dataset, with a numeric attribute for each wave-number,
    and a final nominal attribute for the label (which is the class attribute).

    :param wave_numbers:
                The wave-number to create numeric attributes for.
    :param labels:
                The labels for the nominal class attribute.
    :param capacity:
                The desired initial capacity of the dataset.
    :return:
                The Instances dataset.
    """
    # Create an attribute for each wave-number + one for the label
    attributes = create_attributes(wave_numbers, labels)

    # Create the dataset, and set the label attribute as the class attribute
    dataset = Instances.create_instances("spectra", attributes, capacity)
    dataset.class_index = len(wave_numbers)

    return dataset


def weka_train(
        data_directory: str,
        output_directory: str,
        classifier_classname: str,
        options: str
):
    """
    Trains a WEKA classifier on spectral data.

    :param data_directory:
                The directory to load .spec files from, in sub-dir format.
    :param output_directory:
                The directory to write the serialised classifier to.
    :param classifier_classname:
                The class-name of the classifier to train.
    :param options:
                The options to the classifier.
    """
    with JVMContextManager():
        # Load the .spec files in the data-directory into a dict from
        # filename to (spectrum, label) pairs
        filename_to_spectrum_and_label: Dict[str, Tuple[Spectrum, str]] = {
            os.path.basename(spec_filename): (spectrum, os.path.basename(os.path.split(spec_filename)[0]))
            for spec_filename in glob(data_directory + "/**/*", recursive=True)
            if spec_filename.endswith(".spec")
            for spectrum in reader(spec_filename)
        }

        # Gather a sorted list of all wave-numbers across all spectra
        wave_numbers: List[float] = gather_wave_numbers(
            (spectrum for spectrum, _ in filename_to_spectrum_and_label.values())
        )

        # Gather all the labels
        labels: List[str] = list({
            label for _, label in filename_to_spectrum_and_label.values()
        })

        # Create a reverse-lookup for getting the index of a label
        label_indices: Dict[str, int] = {
            label: index
            for index, label in enumerate(labels)
        }

        # Add each spectrum as an instance of a dataset
        dataset = create_dataset(wave_numbers, labels, len(filename_to_spectrum_and_label))
        for spectrum, label in filename_to_spectrum_and_label.values():
            dataset.add_instance(
                create_instance(
                    wave_numbers,
                    spectrum,
                    label_indices[label]
                )
            )

        # Create and build the classifier
        classifier = Classifier(
            classname=classifier_classname,
            options=shlex.split(options)
        )
        classifier.build_classifier(dataset)

        # Write the classifier to disk
        classifier.serialize(output_directory + "/model", dataset)
        with open(output_directory + "/labels", "w") as label_file:
            label_file.write("\n".join(labels))


def weka_predict(
        model_directory: str,
        input_directory: str,
        output_directory: str
):
    """
    Uses a WEKA classifier to predict labels for spectral data.

    :param model_directory:
                The directory to load the serialised classifier from.
    :param input_directory:
                The directory of .spec files to classify.
    :param output_directory:
                The directory to write the classifications into.
    """
    with JVMContextManager():
        # Read in labels
        with open(model_directory + "/labels", "r") as label_file:
            labels = label_file.read().splitlines()

        # Load the classifier as a filter which replaces the class attribute with the predictions
        classifier = Filter(
            classname="weka.filters.supervised.attribute.AddClassification",
            options=[
                "-serialized", model_directory + "/model",
                "-classification",
                "-remove-old-class"
            ]
        )

        # Load the .spec files in the input-directory into a dict from
        # filename to spectrum
        filename_to_spectrum: Dict[str, Spectrum] = {
            os.path.basename(spec_filename): spectrum
            for spec_filename in glob(input_directory + "/**/*", recursive=True)
            if spec_filename.endswith(".spec")
            for spectrum in reader(spec_filename)
        }

        # Gather a sorted list of all wave-numbers across all spectra
        wave_numbers: List[float] = gather_wave_numbers(
            (spectrum for spectrum in filename_to_spectrum.values())
        )

        # Add each spectrum as an instance of a dataset
        dataset = create_dataset(wave_numbers, labels, len(filename_to_spectrum))
        for spectrum in filename_to_spectrum.values():
            dataset.add_instance(
                create_instance(
                    wave_numbers,
                    spectrum,
                    None
                )
            )

        # Applied the filter to the dataset
        classifier.inputformat(dataset)
        classified: Instances = classifier.filter(dataset)

        # Write the classifications to disk
        for filename, instance in zip(filename_to_spectrum.keys(), classified):
            # Get the classifier's classification (WEKA internally stores the index
            # as a float)
            label_index_float = instance.get_value(instance.class_index)

            # Check a prediction was made
            if math.isnan(label_index_float):
                continue

            # Get the corresponding label
            label = labels[round(label_index_float)]

            # Write the label to disk
            with open(output_directory + "/" + filename.replace(".spec", ".txt"), "w") as pred_file:
                pred_file.write(label)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="method")
    train_parser = sub.add_parser("train")
    train_parser.add_argument(
        dest="data_directory",
        help="Dataset directory.",
        metavar="DATA_DIR"
    )
    train_parser.add_argument(
        dest="output_directory",
        help="Output directory.",
        metavar="OUTPUT_DIR"
    )
    train_parser.add_argument(
        dest="classifier_classname",
        help="Classifier class-name.",
        metavar="CLS"
    )
    train_parser.add_argument(
        dest="options",
        help="Classifier options.",
        metavar="OPT"
    )
    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument(
        dest="model_directory",
        help="Model directory.",
        metavar="MODEL_DIR"
    )
    predict_parser.add_argument(
        dest="input_directory",
        help="Input directory.",
        metavar="INPUT_DIR"
    )
    predict_parser.add_argument(
        dest="output_directory",
        help="Output directory.",
        metavar="OUTPUT_DIR"
    )
    namespace = parser.parse_args()
    if namespace.method == "train":
        weka_train(
            namespace.data_directory,
            namespace.output_directory,
            namespace.classifier_classname,
            namespace.options
        )
    elif namespace.method == "predict":
        weka_predict(
            namespace.model_directory,
            namespace.input_directory,
            namespace.output_directory
        )
