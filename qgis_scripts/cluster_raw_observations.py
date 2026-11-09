"""
Model exported as python.
Name : cluster_animals
Group : 
With QGIS : 33404
"""

from qgis.core import QgsProcessing
from qgis.core import QgsProcessingAlgorithm
from qgis.core import QgsProcessingMultiStepFeedback
from qgis.core import QgsProcessingParameterVectorLayer
from qgis.core import QgsProcessingParameterFeatureSink
import processing


class Clusterrawobs(QgsProcessingAlgorithm):

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer('raw_observations', 'Raw Observations', types=[QgsProcessing.TypeVectorPoint], defaultValue=None))
        self.addParameter(QgsProcessingParameterFeatureSink('Clusters', 'Clusters', type=QgsProcessing.TypeVectorPoint, createByDefault=True, supportsAppend=True, defaultValue='TEMPORARY_OUTPUT'))

    def processAlgorithm(self, parameters, context, model_feedback):
        # Use a multi-step feedback, so that individual child algorithm progress reports are adjusted for the
        # overall progress through the model
        feedback = QgsProcessingMultiStepFeedback(3, model_feedback)
        results = {}
        outputs = {}

        # Collect geometries
        alg_params = {
            'FIELD': ['type','id'],
            'INPUT': parameters['raw_observations'],
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['CollectGeometries'] = processing.run('native:collect', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}

        # Refactor fields
        alg_params = {
            'FIELDS_MAPPING': [{'alias': '','comment': '','expression': '"type"','length': 0,'name': 'type','precision': 0,'sub_type': 0,'type': 10,'type_name': 'text'},{'alias': '','comment': '','expression': '"id"','length': 0,'name': 'id','precision': 0,'sub_type': 0,'type': 4,'type_name': 'int8'}],
            'INPUT': outputs['CollectGeometries']['OUTPUT'],
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['RefactorFields'] = processing.run('native:refactorfields', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}

        # Centroids
        alg_params = {
            'ALL_PARTS': False,
            'INPUT': outputs['RefactorFields']['OUTPUT'],
            'OUTPUT': parameters['Clusters']
        }
        outputs['Centroids'] = processing.run('native:centroids', alg_params, context=context, feedback=feedback, is_child_algorithm=True)
        results['Clusters'] = outputs['Centroids']['OUTPUT']
        return results

    def name(self):
        return 'ClusterRawObs'

    def displayName(self):
        return 'ClusterRawObs'

    def group(self):
        return ''

    def groupId(self):
        return ''

    def createInstance(self):
        return Clusterrawobs()
