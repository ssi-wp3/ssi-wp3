from typing import Dict, Callable, Optional
from abc import ABC, abstractmethod
import pandas as pd


class ProcessingTask:
    def __init__(self,
                 input_filename: str,
                 output_filename: str,
                 processing_function: Callable[[pd.DataFrame], pd.DataFrame],
                 name: Optional[str] = None
                 ):
        self.__input_filename = input_filename
        self.__output_filename = output_filename
        self.__processing_function = processing_function
        self.__name = processing_function.__name__ if not name else name

    @property
    def name(self) -> str:
        return self.__name

    @property
    def input_filename(self) -> str:
        return self.__input_filename

    @property
    def output_filename(self) -> str:
        return self.__output_filename

    @property
    def processing_function(self) -> Callable[[pd.DataFrame], pd.DataFrame]:
        return self.__processing_function

    def __call__(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # Retrieve only keyword arguments that are in the signature of the processing function
        function_kwargs = {key: value for key, value in kwargs.items()
                           if key in self.__processing_function.__code__.co_varnames}
        return self.__processing_function(data, **function_kwargs)


class Processing:
    def __init__(self):
        self.__processing_tasks = dict()

    @property
    def processing_tasks(self) -> Dict[str, ProcessingTask]:
        return self.__processing_tasks

    @processing_tasks.setter
    def processing_task(self, value: Dict[str, ProcessingTask]):
        self.__processing_tasks = value

    @property
    def input_filenames(self) -> Dict[str, str]:
        return {name: task.input_filename for name, task in self.__processing_tasks.items()}

    @property
    def output_filenames(self) -> Dict[str, str]:
        return {name: task.output_filename for name, task in self.__processing_tasks.items()}

    @abstractmethod
    def get_output_filename(self, function_name: str) -> str:
        pass

    def log_analysis_status(self, function_name):
        print(
            f"Running {function_name} for {self.store_name}")

    def add_processing_function(self,
                                function: Callable,
                                input_filename: str,
                                output_filename: Optional[str] = None,
                                name: str = None):
        """ Add a processing function to the processing_functions dictionary

        Parameters
        ----------
        function : Callable
            The processing function to add

        name : str, optional
            The name of the processing function. If None, the name of the function is used.
        """
        if name is None:
            name = function.__name__
        output_filename = output_filename if output_filename else self.get_output_filename(
            name)

        self.add_processing_task(ProcessingTask(
            input_filename=input_filename,
            output_filename=output_filename,
            processing_function=function,
            name=name
        )
        )

    def add_processing_task(self, processing_task: ProcessingTask):
        """ Add a processing task to the processing_tasks dictionary

        Parameters
        ----------
        processing_task : ProcessingTask
            The processing task to add
        """
        self.__processing_tasks[processing_task.name] = processing_task

    def run(self,
            processing_task: ProcessingTask,
            data: pd.DataFrame,
            **kwargs) -> pd.DataFrame:
        return processing_task(data, **kwargs)

    def run_all(self,
                data: pd.DataFrame,
                **kwargs
                ) -> Dict[str, pd.DataFrame]:
        return {name: self.run(task, data, **kwargs) for name, task in self.__processing_tasks.items()}
