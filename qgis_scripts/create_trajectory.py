"""
Model exported as python.
Name : CreateTrajectory
Group : 
With QGIS : 33404
"""

from qgis.core import QgsProcessing
from qgis.core import QgsProcessingAlgorithm
from qgis.core import QgsProcessingMultiStepFeedback
from qgis.core import QgsProcessingParameterCrs
from qgis.core import QgsProcessingParameterField
from qgis.core import QgsProcessingParameterVectorLayer
from qgis.core import QgsProcessingParameterFeatureSink
import processing


class CreateTrajectory(QgsProcessingAlgorithm):

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterCrs('crs', 'CRS', defaultValue='EPSG:25837'))
        self.addParameter(QgsProcessingParameterField('x', 'X', type=QgsProcessingParameterField.Any, parentLayerParameterName='signs', allowMultiple=False, defaultValue=None))
        self.addParameter(QgsProcessingParameterField('y', 'Y', type=QgsProcessingParameterField.Any, parentLayerParameterName='signs', allowMultiple=False, defaultValue=None))
        self.addParameter(QgsProcessingParameterVectorLayer('InputTrajectory', 'InputTrajectory', types=[QgsProcessing.TypeVectorPoint], defaultValue=None))
        self.addParameter(QgsProcessingParameterFeatureSink('OutputTrajectory', 'OutputTrajectory', type=QgsProcessing.TypeVectorAnyGeometry, createByDefault=True, supportsAppend=True, defaultValue=None))

    def processAlgorithm(self, parameters, context, model_feedback):
        # Use a multi-step feedback, so that individual child algorithm progress reports are adjusted for the
        # overall progress through the model
        feedback = QgsProcessingMultiStepFeedback(3, model_feedback)
        results = {}
        outputs = {}

        # Create points layer from table
        alg_params = {
            'INPUT': parameters['InputTrajectory'],
            'MFIELD': '',
            'TARGET_CRS': parameters['crs'],
            'XFIELD': parameters['x'],
            'YFIELD': parameters['y'],
            'ZFIELD': '',
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['CreatePointsLayerFromTable'] = processing.run('native:createpointslayerfromtable', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}

        # Delete duplicate geometries
        alg_params = {
            'INPUT': outputs['CreatePointsLayerFromTable']['OUTPUT'],
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['DeleteDuplicateGeometries'] = processing.run('native:deleteduplicategeometries', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}

        # Refactor fields
        alg_params = {
            'FIELDS_MAPPING': [{'alias': '','comment': '','expression': '"fid"','length': 0,'name': 'fid','precision': 0,'sub_type': 0,'type': 4,'type_name': 'int8'},{'alias': '','comment': '','expression': '"Image"','length': 0,'name': 'Image','precision': 0,'sub_type': 0,'type': 4,'type_name': 'int8'},{'alias': '','comment': '','expression': '"X"','length': 0,'name': 'X','precision': 0,'sub_type': 0,'type': 6,'type_name': 'double precision'},{'alias': '','comment': '','expression': '"Y"','length': 0,'name': 'Y','precision': 0,'sub_type': 0,'type': 6,'type_name': 'double precision'},{'alias': '','comment': '','expression': '"Z"','length': 0,'name': 'Z','precision': 0,'sub_type': 0,'type': 6,'type_name': 'double precision'},{'alias': '','comment': '','expression': '"Omega"','length': 0,'name': 'Omega','precision': 0,'sub_type': 0,'type': 6,'type_name': 'double precision'},{'alias': '','comment': '','expression': '"Phi"','length': 0,'name': 'Phi','precision': 0,'sub_type': 0,'type': 6,'type_name': 'double precision'},{'alias': '','comment': '','expression': '"Kappa"','length': 0,'name': 'Kappa','precision': 0,'sub_type': 0,'type': 6,'type_name': 'double precision'}],
            'INPUT': outputs['DeleteDuplicateGeometries']['OUTPUT'],
            'OUTPUT': parameters['OutputTrajectory']
        }
        outputs['RefactorFields'] = processing.run('native:refactorfields', alg_params, context=context, feedback=feedback, is_child_algorithm=True)
        results['OutputTrajectory'] = outputs['RefactorFields']['OUTPUT']
        return results

    def name(self):
        return 'CreateTrajectory'

    def displayName(self):
        return 'CreateTrajectory'

    def group(self):
        return ''

    def groupId(self):
        return ''

    def createInstance(self):
        return CreateTrajectory()
