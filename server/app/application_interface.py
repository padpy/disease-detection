from abc import ABC, abstractmethod

class ApplicationInterface(ABC):
    @abstractmethod
    def segment_plant(self, image, task='leaf', data=None):
        pass

    @abstractmethod
    def plant_status(self, plant_id):
        pass

    @abstractmethod
    def plant_data(self, plant_id):
        pass

    @abstractmethod
    def get_image(self ,plant_id, image_name):
        pass

    @abstractmethod
    def get_plant_ids(self):
        pass

    @abstractmethod
    def get_trials(self):
        pass

    @abstractmethod
    def create_trial(self, trial_data):
        pass
