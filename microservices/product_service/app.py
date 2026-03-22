# write the crud api for products service and run on port 8000

import fastapi
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 

class Product(BaseModel):
    id: int
    name: str
    price: float
    description: str
    stock: int
products = []

@app.post("/products/", response_model=Product)
def create_product(product: Product):
    products.append(product)
    return product

@app.get("/products/", response_model=List[Product])
def read_products():
    return products

@app.get("/products/{product_id}", response_model=Product)
def read_product(product_id: int):
    for product in products:
        if product.id == product_id:
            return product
        
    raise HTTPException(status_code=404, detail="Product not found")

# Update a product by ID
@app.put("/products/{product_id}", response_model=Product)
def update_product(product_id: int, updated_product: Product):
    for index, product in enumerate(products):
        if product.id == product_id:
            products[index] = updated_product
            return products[index]
    raise HTTPException(status_code=404, detail="Product not found")


@app.delete("/products/{product_id}")
def delete_product(product_id: int):
    for index, product in enumerate(products):
        if product.id == product_id:
            del products[index]
            return {"detail": "Product deleted"}
    raise HTTPException(status_code=404, detail="Product not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)