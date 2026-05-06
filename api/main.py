import sys
import os
import io
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.engine import DataRefinery

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.post("/process")
async def process_data(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        refinery = DataRefinery(df)
        result = refinery.process()
        
        df_refined = result["refined_data"]
        insights = [
            f"Input Dimensions: {len(df)} rows x {len(df.columns)} original columns",
            f"Final Dimensions: {len(df_refined)} rows x {len(df_refined.columns)} optimized columns",
            f"Transformation: Applied Yeo-Johnson power transform for Gaussian distribution."
        ]

        return {
            "status": "success",
            "logs": result["logs"],
            "insights": insights,
            "raw_preview": df.head(10).replace([np.nan], None).to_dict(orient="records"),
            "refined_preview": df_refined.head(10).to_dict(orient="records"),
            "full_refined": df_refined.to_dict(orient="records")
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)