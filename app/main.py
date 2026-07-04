import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.import_routes import router as import_router
from app.api.routes.query_routes import router as query_router
from app.api.routes.eval_routes import router as eval_router
from app.api.routes.kb_routes import router as kb_router

app = FastAPI(
    title="KAAS",
    description="知识库文件导入 + 智能问答服务"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(import_router)
app.include_router(query_router)
app.include_router(eval_router)
app.include_router(kb_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
