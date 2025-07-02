import pkgutil
import inspect
from .base import BaseOCRService

# Dictionary to hold all registered OCR services
OCR_SERVICES = {}

def register_service(service_class):
    """
    Registers an OCR service class in the OCR_SERVICES dictionary.
    """
    if issubclass(service_class, BaseOCRService) and service_class is not BaseOCRService:
        service_name = getattr(service_class, '_service_name', None)
        if service_name and service_name != "base":
            OCR_SERVICES[service_name] = service_class()
            print(f"Registered OCR service: {service_name}")

def discover_services():
    """
    Dynamically discovers and registers OCR services from this package.
    """
    package_path = __path__
    package_name = __name__

    for _, module_name, _ in pkgutil.iter_modules(package_path):
        # Import the module
        module = __import__(f"{package_name}.{module_name}", fromlist=["*"])
        
        # Find all classes in the module and register them if they are a service
        for name, obj in inspect.getmembers(module, inspect.isclass):
            register_service(obj)

# Discover and register services when the package is imported
discover_services()

def get_service(service_name: str) -> BaseOCRService:
    """
    Returns an instance of the requested OCR service.
    """
    service = OCR_SERVICES.get(service_name)
    if not service:
        raise ValueError(f"OCR service '{service_name}' not found.")
    return service