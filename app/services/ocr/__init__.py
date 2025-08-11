import pkgutil
import inspect
from .base import BaseOCRService

# Registries
# - OCR_SERVICE_CLASSES: discovered service classes (name -> class)
# - OCR_SERVICES: instantiated singletons (name -> instance), created lazily
OCR_SERVICE_CLASSES = {}
OCR_SERVICES = {}

def register_service(service_class):
    """
    Register an OCR service CLASS in OCR_SERVICE_CLASSES.
    Do NOT instantiate here to avoid heavy model loads at import time.
    """
    if inspect.isclass(service_class) and issubclass(service_class, BaseOCRService) and service_class is not BaseOCRService:
        service_name = getattr(service_class, '_service_name', None)
        if service_name and service_name != "base":
            OCR_SERVICE_CLASSES[service_name] = service_class

def discover_services():
    """
    Dynamically discovers and registers OCR services from this package.
    """
    package_path = __path__
    package_name = __name__

    for _, module_name, _ in pkgutil.iter_modules(package_path):
        try:
            # Import the module
            module = __import__(f"{package_name}.{module_name}", fromlist=["*"])
            
            # Find all classes in the module and register them if they are a service
            for name, obj in inspect.getmembers(module, inspect.isclass):
                register_service(obj)
        except ImportError as e:
            # Skip modules that can't be imported (e.g., marker during build phase)
            pass
        except Exception as e:
            # Log other errors but continue
            pass

# Discover and register services when the package is imported
discover_services()

def get_service(service_name: str) -> BaseOCRService:
    """
    Returns a singleton instance of the requested OCR service, creating it lazily.
    """
    if service_name in OCR_SERVICES:
        return OCR_SERVICES[service_name]
    service_class = OCR_SERVICE_CLASSES.get(service_name)
    if not service_class:
        raise ValueError(f"OCR service '{service_name}' not found.")
    try:
        instance = service_class()
        OCR_SERVICES[service_name] = instance
        return instance
    except Exception as e:
        # Surface a clear error rather than hiding the service
        raise ValueError(f"Failed to initialize OCR service '{service_name}': {e}")

def service_names():
    """
    Return the list of discovered OCR service names.
    """
    return list(OCR_SERVICE_CLASSES.keys())