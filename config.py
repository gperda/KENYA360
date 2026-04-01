
import configparser
import os
import Metashape

class Config:
    def __init__(self, config_file):
        opts = configparser.ConfigParser()
        opts.read(config_file)
        self.input_dir = opts.get("General", "input_dir", fallback=None)
        if self.input_dir is None:
            raise ValueError("Missing required config parameter: [General] input_dir")
        if not os.path.exists(self.input_dir):
            raise ValueError(f"Input directory does not exist: {self.input_dir}")

        self.frames_dir = os.path.join(self.input_dir, "frames")
        if not os.path.exists(self.frames_dir):
            raise ValueError(f"Images directory does not exist: {self.frames_dir}")
        self.output_dir = opts.get("General", "output_dir", fallback=None).strip()
        if self.output_dir is None:
            raise ValueError("Missing required config parameter: [General] output_dir")
                
        self.signs_detection_dir = os.path.join(self.output_dir, "detection/animals")
        self.signs_positions_file = os.path.join(self.signs_detection_dir, "animals_positions.csv")
        self.metashape_dir = os.path.join(self.input_dir, "metashape")
        self.maps_dir = os.path.join(self.output_dir, "maps")
        self.doc = Metashape.Document()
        self.sam3_half = opts.getboolean("Detection", "sam3_half", fallback=False)
        
        model = opts.get("Detection", "model", fallback=None)
        self.model = model
        if (model == "yolo"):
            self.model_path = opts.get("Detection", "model_path", fallback=None)
            self.model_confidence = opts.getfloat("Detection", "model_confidence", fallback=0.2)
            if self.model_confidence < 0 or self.model_confidence > 1:
                self.model_confidence = 0.2
        elif (model == "sam3"):
            self.model_path = None
            self.model_confidence = None
        elif (model == "sahi"):
            self.model_path = opts.get("Detection", "model_path", fallback=None)
        else:
            raise ValueError(f"Unsupported detection model specified in config: {model}")
       
        self.crs = opts.get("Mapping", "crs", fallback="EPSG:4326")
        # self.block_size = opts.getint("Stereo", "block_size", fallback = 500)


    def get(self, section, param, fallback=None):
        """Get a parameter from a section, with optional fallback."""
        return self._config.get(section, param, fallback=fallback)

    def get_section(self, section):
        """Get all parameters from a section as a dict."""
        if section in self._config:
            return dict(self._config[section])
        return {}

    def sections(self):
        """Return a list of section names."""
        return self._config.sections()