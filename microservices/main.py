import subprocess
import sys
import os

# Service configurations: [command, description, working_directory]
services = [
    {
        "cmd": [sys.executable, "-m", "uvicorn", "calculator_service.app:app", "--port", "8003"],
        "name": "Calculator Service",
        "port": 8003,
        "description": "Basic arithmetic operations API",
        "cwd": None
    },
    {
        "cmd": [sys.executable, "-m", "uvicorn", "product_service.app:app", "--port", "8000"],
        "name": "Product Service",
        "port": 8000,
        "description": "Product CRUD operations API",
        "cwd": None
    },
    {
        "cmd": [sys.executable, "-m", "uvicorn", "todo_service.app:app", "--port", "8001"],
        "name": "Todo Service",
        "port": 8001,
        "description": "Task management API",
        "cwd": None
    },
    {
        "cmd": [sys.executable, "-m", "uvicorn", "main:app", "--port", "8005"],
        "name": "ML Services",
        "port": 8005,
        "description": "Machine Learning APIs (Rain Prediction, Car Evaluation, Social Media Clustering)",
        "cwd": "ml_services"
    }
]

processes = []

print("=" * 80)
print("STARTING ALL MICROSERVICES")
print("=" * 80)

try:
    for service in services:
        print(f"\n> Starting: {service['name']}")
        print(f"   Port: {service['port']}")
        print(f"   Description: {service['description']}")
        print(f"   Command: {' '.join(service['cmd'])}")

        # Set working directory if specified
        cwd = os.path.join(os.path.dirname(__file__), service['cwd']) if service['cwd'] else None
        p = subprocess.Popen(service['cmd'], cwd=cwd)
        processes.append(p)

    print("\n" + "=" * 80)
    print("ALL SERVICES STARTED SUCCESSFULLY!")
    print("=" * 80)
    print("\nService URLs:")
    print(f"  - Calculator Service:  http://localhost:8003")
    print(f"  - Product Service:     http://localhost:8000")
    print(f"  - Todo Service:        http://localhost:8001")
    print(f"  - ML Services:         http://localhost:8005")
    print(f"  - ML Services Docs:    http://localhost:8005/docs")
    print("\nPress CTRL+C to stop all services")
    print("=" * 80 + "\n")

    # Wait for all processes
    for p in processes:
        p.wait()

except KeyboardInterrupt:
    print("\n" + "=" * 80)
    print("STOPPING ALL SERVICES...")
    print("=" * 80)
    for p in processes:
        p.terminate()
    print("\nAll services stopped successfully!")
    sys.exit(0)
