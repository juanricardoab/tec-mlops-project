import pytest
from bikeSharingModel import BikeSharingModel

class TestBikeSharingModel:
    def setup_method(self):
        self.model = BikeSharingModel(275)
        
    def test_check_create_object(self):
        assert self.model.fileNumber == 275