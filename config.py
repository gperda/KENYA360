
import configparser
import os

class Config:
    def __init__(self, config_file):
        opts = configparser.ConfigParser()
        opts.read(config_file)
        self._config = opts
        self.input_dir = opts.get("General", "input_dir", fallback=None)
        if self.input_dir is None:
            raise ValueError("Missing required config parameter: [General] input_dir")
        if not os.path.exists(self.input_dir):
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        camera = opts.get("General", "camera", fallback=None)
        if camera is None:
            raise ValueError("Missing required config parameter: [General] camera")
        self.frames_dir = os.path.join(self.input_dir, camera)
        if not os.path.exists(self.frames_dir):
            raise ValueError(f"Images directory does not exist: {self.frames_dir}")
        self.output_dir = opts.get("General", "output_dir", fallback=None).strip()
        if self.output_dir is None:
            raise ValueError("Missing required config parameter: [General] output_dir")

        self.trajectory_source = opts.get("General", "trajectory_source", fallback="metashape").strip().lower()
        if self.trajectory_source not in {"metashape", "colmap", "slam"}:
            raise ValueError(
                "Unsupported trajectory source specified in config: "
                f"{self.trajectory_source}. Supported values: metashape, colmap, slam"
            )

        self.depth_source = opts.get("General", "depth_source", fallback="metashape").strip().lower()
        if self.depth_source not in {"metashape", "fastfoundation", "colmap", "slam"}:
            raise ValueError(
                "Unsupported depth source specified in config: "
                f"{self.depth_source}. Supported values: metashape, fastfoundation, colmap, slam"
            )

        self.fastfoundation_conda_env = opts.get("General", "fastfoundation_conda_env", fallback=None)
        self.fastfoundation_module = opts.get(
            "General", "fastfoundation_module", fallback="scripts.run_demo_batch"
        )

        prompts = opts.get("Detection", "prompts", fallback=None)
        if prompts is None:
            raise ValueError("Missing required config parameter: [Detection] prompts")

        self.prompts = [prompt.strip() for prompt in prompts.split(",") if prompt.strip()]
        detection_dir = os.path.join(self.output_dir, "detection")
        self.signs_detection_dir = os.path.join(detection_dir, "animals")
        self.signs_positions_file = os.path.join(detection_dir, "animals_positions.csv")
        self.metashape_dir = os.path.join(self.input_dir, "metashape")
        self.maps_dir = os.path.join(self.output_dir, "maps")
        self.raw_observations_file = os.path.join(self.maps_dir, "raw_observations.gpkg")
        self.sam3_half = opts.getboolean("Detection", "sam3_half", fallback=False)
        self.depth_dir = os.path.join(self.output_dir, "depth")
        self.depth_overlay_dir = os.path.join(self.depth_dir, "overlay")


        if self.trajectory_source == "metashape":
            self.trajectory_file_path = os.path.join(self.metashape_dir, "trajectory.csv") 

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
       
        self.crs = opts.get("Mapping", "crs", fallback="EPSG:25837")
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