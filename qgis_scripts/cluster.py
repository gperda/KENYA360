"""
Model exported as python.
Name : cluster_signs
Group : 
With QGIS : 33404
"""

from qgis.core import QgsProcessing
from qgis.core import QgsProcessingAlgorithm
from qgis.core import QgsProcessingMultiStepFeedback
from qgis.core import QgsProcessingParameterVectorLayer
from qgis.core import QgsProcessingParameterFeatureSink
from qgis.core import QgsProcessingParameterField
from qgis.core import QgsProcessingParameterCrs
from qgis.core import QgsProcessingParameterNumber
import processing


class Cluster(QgsProcessingAlgorithm):

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer('signs', 'Signs', types=[QgsProcessing.TypeVectorPoint], defaultValue=None))
        self.addParameter(QgsProcessingParameterField('x', 'X', type=QgsProcessingParameterField.Any, parentLayerParameterName='signs', allowMultiple=False, defaultValue=None))
        self.addParameter(QgsProcessingParameterField('y', 'Y', type=QgsProcessingParameterField.Any, parentLayerParameterName='signs', allowMultiple=False, defaultValue=None))
        self.addParameter(QgsProcessingParameterField('z', 'Z', type=QgsProcessingParameterField.Any, parentLayerParameterName='signs', allowMultiple=False, defaultValue=None))
        self.addParameter(QgsProcessingParameterCrs('crs', 'CRS', defaultValue='EPSG:25832'))
        self.addParameter(QgsProcessingParameterNumber('distance', 'Distance', type=QgsProcessingParameterNumber.Double, defaultValue=None))
        self.addParameter(QgsProcessingParameterFeatureSink('OutputSignsLayer', 'Output Signs Layer', type=QgsProcessing.TypeVectorAnyGeometry, createByDefault=True, supportsAppend=True, defaultValue=None))
    
    def processAlgorithm(self, parameters, context, model_feedback):
        # Use a multi-step feedback, so that individual child algorithm progress reports are adjusted for the
        # overall progress through the model
        feedback = QgsProcessingMultiStepFeedback(6, model_feedback)
        results = {}
        outputs = {}

        # Create points layer from table
        alg_params = {
            'INPUT': parameters['signs'],
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
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['DeleteDuplicateGeometries'] = processing.run('native:deleteduplicategeometries', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}

        # DBSCAN clustering
        alg_params = {
            'DBSCAN*': False,
            'EPS': parameters['distance'],
            'FIELD_NAME': 'CLUSTER_ID',
            'INPUT': outputs['DeleteDuplicateGeometries']['OUTPUT'],
            'MIN_SIZE': 1,
            'SIZE_FIELD_NAME': 'CLUSTER_SIZE',
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['DbscanClustering'] = processing.run('native:dbscanclustering', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(3)
        if feedback.isCanceled():
            return {}

        # Collect geometries
        alg_params = {
            'FIELD': ['type','CLUSTER_ID'],
            'INPUT': outputs['DbscanClustering']['OUTPUT'],
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['CollectGeometries'] = processing.run('native:collect', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(4)
        if feedback.isCanceled():
            return {}

        # Centroids
        alg_params = {
            'ALL_PARTS': False,
            'INPUT': outputs['CollectGeometries']['OUTPUT'],
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['Centroids'] = processing.run('native:centroids', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(5)
        if feedback.isCanceled():
            return {}

        save_params = {
            'INPUT': outputs['Centroids']['OUTPUT'],
            'OUTPUT': parameters['OutputSignsLayer']
        }
        saved = processing.run('native:savefeatures', save_params, context=context, feedback=feedback, is_child_algorithm=True)
        results['OutputSignsLayer'] = saved['OUTPUT']
        return results

    def name(self):
        return 'Cluster'

    def displayName(self):
        return 'Cluster'

    def group(self):
        return ''

    def groupId(self):
        return ''

    def createInstance(self):
        return Cluster()
