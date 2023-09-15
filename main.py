from datetime import datetime, timedelta
from os import lseek
from typing import Optional
from fastapi import FastAPI
import pandas as pd
import uvicorn
from pydantic import BaseModel
from typing import Dict, Union, List, Literal
from fastapi.middleware.cors import CORSMiddleware
import os
import log
from fastapi.responses import JSONResponse, ORJSONResponse
from fastapi import Request, Response
import time
import traceback
import gc
from diffusers.utils import load_image
import numpy as np

import cv2
from PIL import Image
import torch
from diffusers import (
    StableDiffusionPipeline,
    DiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    EulerDiscreteScheduler,
    UniPCMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionUpscalePipeline,
)
from xdog import scribble_xdog,crease64,crease8
from img_util import base64_cv, img_base64, base64_img, resize_image
from pathlib import Path


logger = log.get_logger(os.path.basename(__file__))


class ScribbleConfig(BaseModel):
    prompt: str
    base64str: str=None
    imagenum: int = 1
    width: int = None
    height: int = None
    filepath: str = None
    base64str_ref:str=None
    filepath_ref:str=None


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    start = time.time()
    response = Response("Internal server error", status_code=500)
    try:
        response = await call_next(request)

    except Exception:
        error_data = [
            f"status: {response.status_code}\n",
            f"params: {request.query_params}\n",
            f"path_params: {request.url.path}\n",
            f"time: {time.time() - start}\n",
            f"traceback: {traceback.format_exc()[-2000:]}",
        ]

        error_msg = "".join(error_data)
        logger.error(error_msg)

    end = time.time()
    logger.info(
        f"{request.client.host}:{request.client.port} {request.url.path} {response.status_code} took {round(end-start,5)}"
    )
    return response


@app.get("/")
# @cache(expire=60)
async def read_root():  # 定义根目录方法
    return ORJSONResponse({"result": "Aigc Web Server"})


def init_pipline(pip_name):
    match pip_name:
        case "scribble":
            if (
                not hasattr(app.state, "pipline_type")
                or app.state.pipline_type != "scribble"
            ):
                controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/control_v11p_sd15_scribble",
                    use_safetensors=True,
                    torch_dtype=torch.float16,
                    cache_dir="./models",
                )
                app.state.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                    "./models/realistic_vision/",
                    use_safetensors=True,
                    torch_dtype=torch.float16,
                    controlnet=controlnet,
                )
                app.state.pipeline.scheduler = (
                    EulerAncestralDiscreteScheduler.from_config(
                        app.state.pipeline.scheduler.config
                    )
                )
                app.state.pipline_type = "scribble"
                app.state.pipeline.enable_xformers_memory_efficient_attention()
                app.state.pipeline.enable_attention_slicing()
                app.state.pipeline = app.state.pipeline.to("cuda")
        case "upscaler":
            if (
                not hasattr(app.state, "pipline_type")
                or app.state.pipline_type != "upscaler"
            ):
                app.state.pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                    "./models/models--stabilityai--stable-diffusion-x4-upscaler/snapshots/7b427816f5ce63180d60cc023de189b5d1d0360b",
                    revision="fp16",
                    torch_dtype=torch.float16,
                    cache_dir="./models",
                )
                app.state.pipline_type = "upscaler"
                app.state.pipeline.enable_xformers_memory_efficient_attention()
                app.state.pipeline.enable_attention_slicing()
                app.state.pipeline.enable_model_cpu_offload()
                # app.state.pipeline = app.state.pipeline.to("cuda")
        case "scribble_img2img":
            if (
                not hasattr(app.state, "pipline_type")
                or app.state.pipline_type != "scribble_img2img"
            ):
                controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/control_v11p_sd15_scribble",
                    # 'lllyasviel/sd-controlnet-scribble',
                    use_safetensors=True,
                    torch_dtype=torch.float16,
                    cache_dir="./models",
                )
                app.state.pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                    "./models/realistic_vision/",
                    use_safetensors=True,
                    torch_dtype=torch.float16,
                    controlnet=controlnet,
                    # cache_dir="./models",
                )
                app.state.pipeline.scheduler = UniPCMultistepScheduler.from_config(
                    app.state.pipeline.scheduler.config
                )
                app.state.pipline_type = "scribble_img2img"
                app.state.pipeline.enable_xformers_memory_efficient_attention()
                app.state.pipeline.enable_attention_slicing()
                app.state.pipeline = app.state.pipeline.to("cuda")
        case "diffusion":
            if (
                not hasattr(app.state, "pipline_type")
                or app.state.pipline_type != "diffusion"
            ):
                app.state.pipeline = StableDiffusionPipeline.from_pretrained(
                    "./models/realistic_vision/",
                    use_safetensors=True,
                    torch_dtype=torch.float16,
                    # safety_checker=None
                )
                # app.state.pipeline.scheduler = UniPCMultistepScheduler.from_config(
                #     app.state.pipeline.scheduler.config
                # )
                app.state.pipline_type = "diffusion"
                app.state.pipeline.enable_xformers_memory_efficient_attention()
                app.state.pipeline.enable_attention_slicing()
                app.state.pipeline = app.state.pipeline.to("cuda")


@app.on_event("startup")
async def _startup():
    # init_pipline("scribble")
    logger.debug("startup done")


@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, "pipline_type"):
        del app.state.pipeline
    torch.cuda.empty_cache()
    gc.collect()
    logger.debug("shutdown done")


def scribble_pic(base64str, base64=True,rectange=False,crease=False):
    if base64:
        cv_img = base64_cv(base64str)
    else:
        cv_img = cv2.imread(base64str)
    # print(f'cv_img.shape={cv_img.shape}')
    if rectange:
        height, width, _ = cv_img.shape
        # 计算正方形的大小
        size = min(width, height)

        # 计算截取的起始位置
        x = (width - size) // 2
        y = (height - size) // 2

        # 截取中心正方形
        cv_img = cv_img[y:y+size, x:x+size]
    if crease:
        res = crease64(max(cv_img.shape[0],cv_img.shape[1]))
    else:
        res = 512
    # print(f'res={res}')
    cv_img = scribble_xdog(cv_img, res=res, thr_a=16)
    # img = Image.fromarray(cv_img.astype("uint8"), "RGB")
    img = Image.fromarray(cv_img)
    # print(f'img size={img.size}')
    return img


def prompts_generate(promt, num):
    logger.info(f"promt='{promt}'")
    promt = promt + 'RAW photo, 8k, uhd, dslr, high quality'
    prompts = list([promt] * num)
    negative_prompts = [
        "semi-realistic, cgi, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifact, ugly"
    ] * num
    t = int(round(time.time() * 1000))
    generator = [torch.Generator("cuda").manual_seed(_) for _ in range(t, t + num)]
    return prompts, negative_prompts, generator, [x for x in range(t, t + num)]


def pipline_process(
    prompt: str, img=None, img_ref=None, imagenum=1, width=512, height=512, mode: str = "diffusion"
):
    prompts, negative_prompts, generator, seeds = prompts_generate(prompt, imagenum)
    width,height = crease8(width),crease8(height)
    match mode:
        case "controlnet":
            images = app.state.pipeline(
                prompt=prompts,
                image=img,
                negative_prompt=negative_prompts,
                num_inference_steps=30,
                width=width,
                height=height,
            ).images
        case "diffusion":
            images = app.state.pipeline(
                prompt=prompts,
                negative_prompt=negative_prompts,
                num_inference_steps=30,
                width=width,
                height=height,
            ).images
        case 'controlnet_img2img':
             images = app.state.pipeline(
                prompt=prompts,
                image=img_ref,
                control_image = img,
                negative_prompt=negative_prompts,
                # num_inference_steps=30,
                # width=width,
                # height=height,
            ).images

    return images, seeds

@app.post("/diffusion")
async def diffusion(req: ScribbleConfig):
    init_pipline("diffusion")
    if not req.width:
        width = 512
    if not req.height:
        height = 512
    images, seeds = pipline_process(
        req.prompt, imagenum = req.imagenum, width=width, height=height, mode='diffusion'
    )
    for i in range(len(images)):
        images[i].save(Path(f"results/diffusion/{seeds[i]}.png"))
    return {"images_base64": [img_base64(x) for x in images], "seeds": seeds}

@app.post("/diff_scribble")
async def diff_scribble(req: ScribbleConfig):
    init_pipline("scribble")
    if not req.filepath:
        control_img = scribble_pic(req.base64str, base64=True)
    else:
        control_img = scribble_pic(req.filepath, base64=False)
    control_img.save('./results/scribble_result.png')
    if not req.width:
        width = control_img.size[0]
    if not req.height:
        height = control_img.size[1]
    images, seeds = pipline_process(
        req.prompt, img = control_img, imagenum = req.imagenum, width=width, height=height, mode='controlnet'
    )
    for i in range(len(images)):
        images[i].save(Path(f"results/scribble/{seeds[i]}.png"))
    return {"images_base64": [img_base64(x) for x in images], "seeds": seeds}

@app.post("/diff_scribble_img2img")
async def diff_scribble_img2img(req: ScribbleConfig):
    init_pipline("scribble_img2img")
    if not req.filepath:
        control_img = scribble_pic(req.base64str, base64=True,rectange=True,crease=False)
    else:
        control_img = scribble_pic(req.filepath, base64=False,rectange=True,crease=False)
    control_img.save('./results/scribble_result.png')
    if not req.filepath_ref:
        style_img = base64_img(req.base64str_ref)
    else:
        # style_img = load_image(req.filepath_ref)
        style_img = Image.open(req.filepath_ref).convert("RGB")
        # style_img = np.array(style_img)
    style_img = style_img.resize((control_img.size[0],control_img.size[1]))
    images, seeds = pipline_process(
        req.prompt, img = [control_img]*req.imagenum,img_ref=style_img, imagenum = req.imagenum, width=control_img.size[0], height=control_img.size[1], mode='controlnet_img2img'
    )
    for i in range(len(images)):
        images[i].save(Path(f"results/scribble_img2img/{seeds[i]}.png"))
    return {"images_base64": [img_base64(x) for x in images], "seeds": seeds}


@app.post("/diff_upscaler")
async def diff_upscaler(req: ScribbleConfig):
    init_pipline("upscaler")
    if not req.filepath:
        img = base64_img(req.base64str)
    else:
        img = Image.open(req.filepath).convert("RGB")
    img = resize_image(img)
    upscaled_image = app.state.pipeline(prompt=req.prompt, image=img).images[0]
    upscaled_image.save(Path(f"results/upscaled_image.png"))
    return {"images_base64": [img_base64(upscaled_image)], "seeds": []}


if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=8800, reload=True)
