 
# Todo Service
# Description: Manages simple task tracking and completion status.
# Features:
# Create a todo with a title (default: not done)
# List all todos
# Retrieve a specific todo by ID
# Mark a todo as done
# Delete a todo
# Stores data with fields: id, title, done, created_at

import fastapi
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from datetime import datetime
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Todo(BaseModel):
    id: int
    title: str
    done: bool = False
    created_at: datetime
todos = []

@app.post("/todos/", response_model=Todo)
def create_todo(todo: Todo):
    todos.append(todo)
    return todo

@app.get("/todos/", response_model=List[Todo])
def read_todos():
    return todos



@app.get("/todos/{todo_id}", response_model=Todo)

def read_todo(todo_id: int):
    for todo in todos:
        if todo.id == todo_id:
            return todo
    raise HTTPException(status_code=404, detail="Todo not found")


@app.put("/todos/{todo_id}", response_model=Todo)
def mark_todo_done(todo_id: int, updated_todo: Todo):
    for index, todo in enumerate(todos):
        if todo.id == todo_id:
            todos[index] = updated_todo
            return todos[index]
    raise HTTPException(status_code=404, detail="Todo not found")


@app.delete("/todos/{todo_id}")
def delete_todo(todo_id: int):
    for index, todo in enumerate(todos):
        if todo.id == todo_id:
            del todos[index]
            return {"detail": "Todo deleted"}
    raise HTTPException(status_code=404, detail="Todo not found")


#create the api for marking a todo as done``
@app.patch("/todos/{todo_id}/done", response_model=Todo)
def mark_todo_as_done(todo_id: int):
    for index, todo in enumerate(todos):
        if todo.id == todo_id:
            todos[index].done = True
            return todos[index]
    raise HTTPException(status_code=404, detail="Todo not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)