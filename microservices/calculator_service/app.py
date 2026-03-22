# Calculator Service
# Description: Provides basic arithmetic operations through simple API endpoints.
# Features:
# Perform addition, subtraction, multiplication, and division
# Accept inputs via query parameters
# Return result along with operation description
# Maintain in-memory history of the last 10 operations
# Handle divide-by-zero errors with proper HTTP error response




from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Operands(BaseModel):
    a: float
    b: float
history = []

@app.get("/")
def read_root():
    return {"message": "Welcome to the Calculator Service!"}


@app.post("/add")
def add(operands: Operands):
    result = operands.a + operands.b
    history.append({"a": operands.a, "b": operands.b, "operation": "+", "result": result})
    if len(history) > 10:
        history.pop(0)
    return {"operation": "addition", "result": result}


@app.post("/subtract")
def subtract(operands: Operands):
    result = operands.a - operands.b
    history.append({"a": operands.a, "b": operands.b, "operation": "-", "result": result})
    if len(history) > 10:
        history.pop(0)
    return {"operation": "subtraction", "result": result}

# Handle multiplication-by-zero errors with proper HTTP error response
@app.post('/multiplication')
def multiplication(operands: Operands):

    if operands.b == 0:
        raise HTTPException(status_code=400, detail="Error: Multiplication by zero is not allowed.")
    
    if operands.a == 0:
        raise HTTPException(status_code=400, detail="Error: First operand cannot be zero.")

    result = operands.a * operands.b
    history.append({"a": operands.a, "b": operands.b, "operation": "×", "result": result})
    if len(history) > 10:
        history.pop(0)
    return {"operation":"multiplication", "result":result}


# Handle divide-by-zero and a is also not llowed to enter 0 errors with proper HTTP error response
@app.post('/division')
def division(operands: Operands):

    if operands.b == 0:
        raise HTTPException(status_code=400, detail="Error: Division by zero is not allowed.")

    if operands.a == 0:
        raise HTTPException(status_code=400, detail="Error: First operand cannot be zero.")

    result = operands.a / operands.b
    history.append({"a": operands.a, "b": operands.b, "operation": "÷", "result": result})
    if len(history) > 10:
        history.pop(0)
    return {"operation":"division", "result":result}


@app.get("/history")
def get_history():
    return {"history": history} 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)


