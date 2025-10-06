import React from 'react';
import { CameraIcon, CpuChipIcon, EyeIcon, PuzzlePieceIcon, ArrowsPointingInIcon, CubeTransparentIcon } from './icons';

const pipelineSteps = [
    {
        title: "Data Acquisition & Annotation",
        description: "Images are collected from an RGB-D camera and annotated with bounding boxes for objects of interest.",
        icon: <CameraIcon />,
    },
    {
        title: "YOLO Model Training",
        description: "The YOLOv8 model is trained on the custom dataset to accurately detect specific objects in real-time.",
        icon: <CpuChipIcon />,
    },
    {
        title: "Model Inference",
        description: "The trained YOLO model is deployed on a live RGB stream to identify and locate objects within the camera's view.",
        icon: <EyeIcon />,
    },
    {
        title: "FastSAM Application",
        description: "Bounding boxes from YOLO are fed into FastSAM to generate precise, pixel-wise segmentation masks for each detected object.",
        icon: <ArrowsPointingInIcon />,
    },
    {
        title: "RGB and Depth Matching",
        description: "The 2D segmentation mask is aligned with the camera's depth map using intrinsic and extrinsic matrix transformations.",
        icon: <PuzzlePieceIcon />,
    },
    {
        title: "3D Reconstruction",
        description: "A 3D point cloud of the segmented object is generated, denoised, and reconstructed, isolating it in 3D space.",
        icon: <CubeTransparentIcon />,
    },
];

const Pipeline: React.FC = () => {
    return (
        <section id="pipeline" className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center">
                <h2 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">The FusionVision Pipeline</h2>
                <p className="mt-4 text-lg text-gray-600">
                    A six-step process from image capture to accurate 3D object reconstruction.
                </p>
            </div>

            <div className="mt-12">
                <img
                    src="https://github.com/safouaneelg/FusionVision/blob/main/images/FusionVisionPipeline.gif?raw=true"
                    alt="FusionVision Pipeline Diagram"
                    className="rounded-lg shadow-2xl mx-auto border border-gray-200"
                />
            </div>

            <div className="mt-20 grid gap-8 md:grid-cols-2 lg:grid-cols-3">
                {pipelineSteps.map((step, index) => (
                    <div key={index} className="bg-white p-6 rounded-lg border border-gray-200 shadow-lg hover:border-cyan-400 hover:shadow-cyan-500/10 transition-all duration-300">
                        <div className="flex items-center gap-4">
                            <div className="flex-shrink-0 h-12 w-12 flex items-center justify-center rounded-lg bg-cyan-500 text-white">
                                {step.icon}
                            </div>
                            <h3 className="text-lg font-bold text-gray-900">{step.title}</h3>
                        </div>
                        <p className="mt-4 text-gray-600">
                            {step.description}
                        </p>
                    </div>
                ))}
            </div>
        </section>
    );
};

export default Pipeline;