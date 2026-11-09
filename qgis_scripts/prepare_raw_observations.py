"""
Model exported as python.
Name : PrepareRawObs
Group : 
With QGIS : 33404
"""

from qgis.core import QgsProcessing
from qgis.gui import QgsFileWidget
from qgis.core import QgsEditorWidgetSetup
from qgis.core import QgsProcessingUtils
from qgis.core import QgsProcessingAlgorithm
from qgis.core import QgsProcessingMultiStepFeedback
from qgis.core import QgsProcessingParameterCrs
from qgis.core import QgsProcessingParameterFile
from qgis.core import QgsProcessingParameterField
from qgis.core import QgsProcessingParameterVectorLayer
from qgis.core import QgsProcessingParameterFeatureSink
import processing


class PrepareRawObs(QgsProcessingAlgorithm):

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterCrs('crs', 'CRS', defaultValue='EPSG:25832'))
        self.addParameter(QgsProcessingParameterFile('folder_path', 'Image folder', behavior=QgsProcessingParameterFile.Folder))
        self.addParameter(QgsProcessingParameterField('x', 'X', type=QgsProcessingParameterField.Any, parentLayerParameterName='animals_detections', allowMultiple=False, defaultValue=None))
        self.addParameter(QgsProcessingParameterField('y', 'Y', type=QgsProcessingParameterField.Any, parentLayerParameterName='animals_detections', allowMultiple=False, defaultValue=None))
        self.addParameter(QgsProcessingParameterField('z', 'Z', type=QgsProcessingParameterField.Any, parentLayerParameterName='animals_detections', allowMultiple=False, defaultValue=None))
        self.addParameter(QgsProcessingParameterVectorLayer('animals_detections', 'Animals detections', types=[QgsProcessing.TypeVectorPoint], defaultValue=None))
        self.addParameter(QgsProcessingParameterFeatureSink('RawObservations', 'Raw observations', type=QgsProcessing.TypeVectorAnyGeometry, createByDefault=True, defaultValue='TEMPORARY_OUTPUT'))

    def processAlgorithm(self, parameters, context, model_feedback):
        # Use a multi-step feedback, so that individual child algorithm progress reports are adjusted for the
        # overall progress through the model
        feedback = QgsProcessingMultiStepFeedback(2, model_feedback)
        results = {}
        outputs = {}

        # Create points layer from table
        alg_params = {
            'INPUT': parameters['animals_detections'],
            'MFIELD': '',
            'TARGET_CRS': parameters['crs'],
            'XFIELD': parameters['x'],
            'YFIELD': parameters['y'],
            'ZFIELD': parameters['z'],
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['CreatePointsLayerFromTable'] = processing.run('native:createpointslayerfromtable', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}

        # Delete duplicate geometries
        alg_params = {
            'INPUT': outputs['CreatePointsLayerFromTable']['OUTPUT'],
            'OUTPUT': parameters['RawObservations']
        }
        outputs['DeleteDuplicateGeometries'] = processing.run('native:deleteduplicategeometries', alg_params, context=context, feedback=feedback, is_child_algorithm=True)
        
        # Attach image to each observation
        output_layer = QgsProcessingUtils.mapLayerFromString(outputs['DeleteDuplicateGeometries']['OUTPUT'], context)
        image_root = self.parameterAsString(parameters, 'folder_path', context)

        image_idx = output_layer.fields().indexFromName('Image') if output_layer else -1
        if output_layer and image_idx != -1:
            updates = {}
            for feature in output_layer.getFeatures():
                image_name = feature[image_idx]
                if image_name is None:
                    continue
                image_name = str(image_name).strip()
                if not image_name:
                    continue

                updates[feature.id()] = {image_idx: f'{image_name}.png'}

            if updates:
                output_layer.dataProvider().changeAttributeValues(updates)

        config = {
            'FileWidget': True,
            'FileWidgetButtonVisible': True,
            'RelativeDefaultPath': 2,
            # 'RelativeStorage': QgsFileWidget.RelativeDefaultPath,
            'DefaultRoot': image_root,
            'DocumentViewer': 1,
            'DocumentViewerHeight': 300,
            'DocumentViewerWidth': 400,
        }

        if output_layer and image_idx != -1:
            setup = QgsEditorWidgetSetup('ExternalResource', config)
            output_layer.setEditorWidgetSetup(image_idx, setup)
        
        results['RawObservations'] = outputs['DeleteDuplicateGeometries']['OUTPUT']
        return results

    def name(self):
        return 'PrepareRawObs'

    def displayName(self):
        return 'PrepareRawObs'

    def group(self):
        return ''

    def groupId(self):
        return ''

    def createInstance(self):
        return PrepareRawObs()
