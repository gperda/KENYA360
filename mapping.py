
from utils import timeit, logger
from qgis.core import QgsApplication, QgsProject, QgsVectorLayer, QgsRasterLayer
from qgis.core import QgsProcessingProvider, QgsProcessingContext, QgsProcessingFeedback, QgsProcessingAlgorithm,QgsCoordinateReferenceSystem
from qgis_scripts.cluster import Cluster
from qgis_scripts.prepare_raw_observations import PrepareRawObs
from qgis_scripts.cluster_raw_observations import Clusterrawobs
from qgis_scripts.create_trajectory import CreateTrajectory
import os
import sys
import shutil

qgis_python = "/usr/share/qgis/python"
qgis_plugins = os.path.join(qgis_python, "plugins")

sys.path.append(qgis_python)
sys.path.append(qgis_plugins)

QgsApplication.setPrefixPath("/usr/lib", True)
import processing

config = None

class AppProvider(QgsProcessingProvider):
    def loadAlgorithms(self):
        self.addAlgorithm(Cluster())
        self.addAlgorithm(PrepareRawObs())
        self.addAlgorithm(Clusterrawobs())
        self.addAlgorithm(CreateTrajectory())

    def id(self, *args, **kwargs):
        """Used for identifying the provider.

        This string should be a unique, short, character only string,
        eg "qgis" or "gdal". This string should not be localised.
        """
        return 'AppProvider'

    def name(self, *args, **kwargs):
        """
        This string should be as short as possible (e.g. "Lastools", not
        "Lastools version 1.0.1 64-bit") and localised.
        """
        return self.tr('AppProvider')

def generate(cfg, execution_times):
    global config 
    config = cfg

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    qgs = QgsApplication([], False)
    qgs.initQgis()

    sys.path.append("./qgis_scripts")
    from processing.core.Processing import Processing
    Processing.initialize()

    provider = AppProvider()
    QgsApplication.processingRegistry().addProvider(provider)

    temp_dir = os.path.join(config.maps_dir, "temp")
    os.makedirs(temp_dir)
    
    # with timeit("mapping.cluster_signs", execution_times):
    #     cluster_signs(config.maps_dir)

    with timeit("mapping.create_trajectory", execution_times):
        create_trajectory(config.maps_dir)

    with timeit("mapping.prepare_row_observations", execution_times):
        prepare_raw_observations(config.maps_dir)

    with timeit("mapping.cluster_raw_observations", execution_times):
        cluster_raw_observations(config.maps_dir)

    shutil.rmtree(temp_dir, ignore_errors=True)

    project = QgsProject.instance()
    project.clear()

    # Add created layers to qgis project
    # vector layers to add to project
    layers = []
    for filename in os.listdir(config.maps_dir):
        path = os.path.join(config.maps_dir, filename)
        layers.append((path, filename, "vector"))

   
    for path, name, layer_type in layers:
        if layer_type == "vector":
            layer = QgsVectorLayer(path, name, "ogr")
            style_path = os.path.join("./qgis_scripts/styles", f"{name.split('.')[0]}.qml")
            if os.path.exists(style_path):
                layer.loadNamedStyle(style_path)
                layer.triggerRepaint()
        else:
            layer = QgsRasterLayer(path, name)
        
        project.addMapLayer(layer)

    project.write(os.path.join(config.maps_dir, config.input_dir + ".qgz"))

    qgs.exitQgis()
    return


def create_heatmap(input):
    """
    Creates point layer for heatmap of percentage of road occupied by cracks.
    """
    feedback = QgsProcessingFeedback()
    context = QgsProcessingContext()

    cracks = os.path.join(config.maps_dir, 'cracks_singlepart.gpkg')
    processing.run("AppProvider:Heatmap", {'cracks_layer':cracks, 'crs':config.crs, 'lane_width':config.lane_width, 'trajectory':config.optimized_trajectory_path, 'transect_length':config.transect_length, 'x':'X', 'y':'Y', 'z':'Z', 'OutputHeatmap':os.path.join(input, 'heatmap.gpkg'), 'OutputVectorLayer':os.path.join(input, 'transects.gpkg')}, context=context, feedback=feedback)
    return

def cluster_signs(input):
    """
    Clean signs detection results by clustering nearby signs of same class
    """
    feedback = QgsProcessingFeedback()
    context = QgsProcessingContext()

    processing.run("AppProvider:Cluster", {'crs':config.crs,'distance':0.4,'signs':config.signs_positions_file,'x':'X','y':'Y','z':'Z','OutputSignsLayer':os.path.join(input, "signs.gpkg")}, context=context, feedback=feedback)
    return

def create_trajectory(input):
    """
    Create trajectory from metashape project
    """
    feedback = QgsProcessingFeedback()
    context = QgsProcessingContext()
    print("MApping with " , config.trajectory_file_path)
    processing.run("AppProvider:CreateTrajectory", {'crs':config.crs,'x': 'X', 'y': 'Y', 'InputTrajectory':config.trajectory_file_path,'OutputTrajectory':os.path.join(input, "trajectory.gpkg")}, context=context, feedback=feedback)
    return

def prepare_raw_observations(input):
    """
    Prepare raw observations for clustering
    """
    feedback = QgsProcessingFeedback()
    context = QgsProcessingContext()

    processing.run("AppProvider:PrepareRawObs", {'crs':config.crs,'x':'X','y':'Y','z':'Z','animals_detections':config.signs_positions_file,'RawObservations':config.raw_observations_file,'folder_path':config.depth_overlay_dir}, context=context, feedback=feedback)
    return

def cluster_raw_observations(input):
    """
    Clean raw observations by clustering nearby observations of same class
    """
    feedback = QgsProcessingFeedback()
    context = QgsProcessingContext()

    processing.run("AppProvider:ClusterRawObs", {'raw_observations':config.raw_observations_file,'Clusters':os.path.join(input, "clusters.gpkg")}, context=context, feedback=feedback)
    return