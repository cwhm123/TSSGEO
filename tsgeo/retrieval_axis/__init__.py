from .discrimination_axis import compute_discrimination_axis, project_onto_axis, project_batch_onto_axis, cosine_similarity_per_layer
from .model import ProbingModel
from .conversation_encoding import ConversationEncoder
from .activation_extraction import RepresentationExtractor, mean_pool_role_spans
from .data_io import Sample, load_samples

__all__ = [
    "compute_discrimination_axis",
    "project_onto_axis",
    "project_batch_onto_axis",
    "cosine_similarity_per_layer",
    "ProbingModel",
    "ConversationEncoder",
    "RepresentationExtractor",
    "mean_pool_role_spans",
    "Sample",
    "load_samples",
]
