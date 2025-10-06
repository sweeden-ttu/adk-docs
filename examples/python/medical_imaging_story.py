#!/usr/bin/env python3
"""
Medical Imaging Story: A Tale of Parallel Discovery
==================================================

This story executes a plan with parallel chapters to create specialized agents
for medical imaging analysis, ultimately producing:
1. A reasoning agent
2. A DICOM vs NIfTI understanding agent  
3. An animated GIF zooming on an aneurysm
4. A PNG rendering a slice of a 4D scan

The story unfolds through parallel chapters that execute asynchronously.
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import nibabel as nib
import pydicom
from io import BytesIO
import zipfile

# Add the kaggle-competition path for medical imaging utilities
sys.path.append('/Users/owner/kaggle-competition')

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types

# Import medical imaging utilities
try:
    from aneurysm_mm_keras.utils.dicom_io import load_dicom_from_zip, convert_to_hu
    from aneurysm_mm_keras.utils.nifti_io import load_nifti_from_zip, ensure_ras_orientation
    from aneurysm_mm_keras.utils.viz import plot_volume_slices, save_volume_as_nifti
except ImportError:
    print("Warning: Medical imaging utilities not found. Using mock implementations.")
    
    def load_dicom_from_zip(zip_file, member_paths):
        """Mock DICOM loader for demonstration."""
        return np.random.rand(64, 64, 64), {"pixel_spacing": [1.0, 1.0], "slice_thickness": 1.0}
    
    def load_nifti_from_zip(zip_file, member_path):
        """Mock NIfTI loader for demonstration."""
        return np.random.rand(64, 64, 64), {"voxel_size": [1.0, 1.0, 1.0]}


class MedicalImagingStory:
    """
    The main story orchestrator that executes parallel chapters
    to create specialized medical imaging agents.
    """
    
    def __init__(self):
        self.app_name = "medical_imaging_story"
        self.user_id = "story_reader"
        self.session_id = "parallel_medical_chapters"
        self.model = "gemini-2.0-flash"
        
        # Initialize session service
        self.session_service = InMemorySessionService()
        self.session = None
        self.results = {}
        
    async def initialize_story(self):
        """Initialize the story environment."""
        print("📖 Initializing Medical Imaging Story...")
        self.session = await self.session_service.create_session(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=self.session_id
        )
        print("✅ Story environment ready!")
    
    def create_reasoning_agent(self) -> LlmAgent:
        """Chapter 1: Create a reasoning agent for medical image analysis."""
        return LlmAgent(
            name="MedicalReasoningAgent",
            model=self.model,
            description="Advanced reasoning agent for medical image analysis and diagnosis",
            instruction="""
            You are an expert medical imaging AI with advanced reasoning capabilities.
            Your expertise includes:
            - Analyzing 3D medical volumes (CT, MRI, CTA, MRA)
            - Detecting anatomical structures and pathologies
            - Reasoning about spatial relationships in medical images
            - Providing differential diagnoses based on imaging findings
            - Understanding medical imaging physics and artifacts
            
            When analyzing medical images, you should:
            1. Systematically examine the image data
            2. Identify key anatomical landmarks
            3. Look for pathological findings
            4. Consider differential diagnoses
            5. Provide evidence-based reasoning for your conclusions
            
            Always maintain medical accuracy and acknowledge limitations in your analysis.
            """,
            tools=[google_search]
        )
    
    def create_dicom_nifti_agent(self) -> LlmAgent:
        """Chapter 2: Create an agent that understands DICOM vs NIfTI differences."""
        return LlmAgent(
            name="DicomNiftiExpertAgent",
            model=self.model,
            description="Specialized agent for understanding DICOM and NIfTI medical imaging formats",
            instruction="""
            You are a medical imaging format specialist with deep knowledge of DICOM and NIfTI standards.
            
            DICOM (Digital Imaging and Communications in Medicine):
            - Standard for medical imaging communication and storage
            - Contains rich metadata (patient info, acquisition parameters, etc.)
            - Each slice is a separate file with headers
            - Supports multiple modalities (CT, MRI, Ultrasound, etc.)
            - Includes patient demographics, study information, and technical parameters
            - Uses tags for metadata organization
            - Can contain 2D slices that need to be assembled into 3D volumes
            
            NIfTI (Neuroimaging Informatics Technology Initiative):
            - Standardized format for neuroimaging data
            - Contains 3D/4D volume data in a single file
            - Includes spatial orientation and transformation matrices
            - Optimized for neuroimaging analysis workflows
            - Supports both .nii and .nii.gz (compressed) formats
            - Includes header with voxel dimensions, orientation, and coordinate system
            - Commonly used in research and analysis pipelines
            
            Key Differences:
            1. DICOM: Multi-file format with rich metadata, NIfTI: Single-file format
            2. DICOM: Clinical workflow focused, NIfTI: Research/analysis focused
            3. DICOM: 2D slices, NIfTI: 3D/4D volumes
            4. DICOM: Patient-centric metadata, NIfTI: Volume-centric metadata
            5. DICOM: Requires assembly, NIfTI: Ready for analysis
            
            You can help users understand when to use each format and how to convert between them.
            """,
            tools=[google_search]
        )
    
    def create_visualization_agent(self) -> LlmAgent:
        """Chapter 3: Create an agent for medical image visualization."""
        return LlmAgent(
            name="MedicalVisualizationAgent",
            model=self.model,
            description="Specialized agent for creating medical image visualizations and animations",
            instruction="""
            You are a medical imaging visualization specialist. Your capabilities include:
            
            Visualization Creation:
            - Generate animated GIFs showing medical image sequences
            - Create static PNG images of medical image slices
            - Produce 3D renderings and volume visualizations
            - Create educational medical imaging content
            
            Animation Techniques:
            - Zoom animations focusing on specific anatomical regions
            - Slice-by-slice progression through 3D volumes
            - Time-lapse sequences for 4D imaging data
            - Highlighting specific anatomical structures or pathologies
            
            Image Processing:
            - Window/level adjustments for optimal contrast
            - Multi-planar reconstruction (MPR)
            - Maximum intensity projection (MIP)
            - Volume rendering techniques
            
            You understand medical imaging physics, anatomy, and the importance of
            maintaining diagnostic quality in visualizations.
            """,
            tools=[google_search]
        )
    
    async def chapter_1_reasoning_agent(self) -> Dict[str, Any]:
        """Execute Chapter 1: Create and test the reasoning agent."""
        print("🧠 Chapter 1: Creating Medical Reasoning Agent...")
        
        reasoning_agent = self.create_reasoning_agent()
        runner = InMemoryRunner(
            agent=reasoning_agent,
            app_name=self.app_name,
            session_service=self.session_service
        )
        
        # Test the reasoning agent
        test_query = """
        Analyze a 3D medical volume for potential intracranial aneurysms. 
        What systematic approach would you take to identify and characterize 
        aneurysms in brain imaging? Include considerations for different 
        imaging modalities (CTA, MRA, MRI) and anatomical locations.
        """
        
        content = types.Content(role="user", parts=[types.Part(text=test_query)])
        
        response_text = ""
        async for event in runner.run_async(
            user_id=self.user_id,
            session_id=self.session_id,
            new_message=content
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        response_text += part.text
        
        result = {
            "agent_name": "MedicalReasoningAgent",
            "capabilities": "Advanced reasoning for medical image analysis",
            "test_response": response_text[:500] + "..." if len(response_text) > 500 else response_text,
            "status": "success"
        }
        
        print("✅ Reasoning Agent created and tested successfully!")
        return result
    
    async def chapter_2_dicom_nifti_agent(self) -> Dict[str, Any]:
        """Execute Chapter 2: Create and test the DICOM/NIfTI understanding agent."""
        print("📁 Chapter 2: Creating DICOM/NIfTI Expert Agent...")
        
        dicom_nifti_agent = self.create_dicom_nifti_agent()
        runner = InMemoryRunner(
            agent=dicom_nifti_agent,
            app_name=self.app_name,
            session_service=self.session_service
        )
        
        # Test the DICOM/NIfTI agent
        test_query = """
        Explain the key differences between DICOM and NIfTI formats for medical imaging.
        When would you choose DICOM over NIfTI, and vice versa? Include practical 
        considerations for clinical workflows and research applications.
        """
        
        content = types.Content(role="user", parts=[types.Part(text=test_query)])
        
        response_text = ""
        async for event in runner.run_async(
            user_id=self.user_id,
            session_id=self.session_id,
            new_message=content
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        response_text += part.text
        
        result = {
            "agent_name": "DicomNiftiExpertAgent",
            "capabilities": "Expert knowledge of DICOM and NIfTI formats",
            "test_response": response_text[:500] + "..." if len(response_text) > 500 else response_text,
            "status": "success"
        }
        
        print("✅ DICOM/NIfTI Agent created and tested successfully!")
        return result
    
    async def chapter_3_visualization_agent(self) -> Dict[str, Any]:
        """Execute Chapter 3: Create and test the visualization agent."""
        print("🎨 Chapter 3: Creating Medical Visualization Agent...")
        
        viz_agent = self.create_visualization_agent()
        runner = InMemoryRunner(
            agent=viz_agent,
            app_name=self.app_name,
            session_service=self.session_service
        )
        
        # Test the visualization agent
        test_query = """
        Design a visualization strategy for an intracranial aneurysm case study.
        Include plans for creating an animated GIF that zooms in on the aneurysm
        and a static PNG showing a key slice. What technical considerations
        are important for medical image visualization?
        """
        
        content = types.Content(role="user", parts=[types.Part(text=test_query)])
        
        response_text = ""
        async for event in runner.run_async(
            user_id=self.user_id,
            session_id=self.session_id,
            new_message=content
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        response_text += part.text
        
        result = {
            "agent_name": "MedicalVisualizationAgent",
            "capabilities": "Medical image visualization and animation",
            "test_response": response_text[:500] + "..." if len(response_text) > 500 else response_text,
            "status": "success"
        }
        
        print("✅ Visualization Agent created and tested successfully!")
        return result
    
    async def chapter_4_create_aneurysm_animation(self) -> Dict[str, Any]:
        """Execute Chapter 4: Create animated GIF zooming on aneurysm."""
        print("🎬 Chapter 4: Creating Aneurysm Animation...")
        
        try:
            # Create synthetic 3D volume with simulated aneurysm
            volume_shape = (64, 64, 64)
            volume = np.random.rand(*volume_shape) * 0.3
            
            # Add simulated aneurysm (spherical structure)
            center = (32, 32, 32)
            radius = 8
            
            # Create aneurysm structure
            y, x, z = np.ogrid[:volume_shape[0], :volume_shape[1], :volume_shape[2]]
            distance = np.sqrt((x - center[1])**2 + (y - center[0])**2 + (z - center[2])**2)
            aneurysm_mask = distance <= radius
            
            # Add aneurysm with higher intensity
            volume[aneurysm_mask] = 0.8
            
            # Create zoom animation
            fig, ax = plt.subplots(figsize=(8, 8))
            
            def animate(frame):
                ax.clear()
                
                # Calculate zoom factor (starts at 1.0, zooms to 3.0)
                zoom_factor = 1.0 + (frame / 30.0) * 2.0
                
                # Calculate crop region for zoom
                crop_size = int(64 / zoom_factor)
                start = (64 - crop_size) // 2
                end = start + crop_size
                
                # Extract and display middle slice
                middle_slice = volume[32, start:end, start:end]
                
                ax.imshow(middle_slice, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'Aneurysm Zoom Animation - Frame {frame+1}/30')
                ax.axis('off')
            
            # Create animation
            anim = animation.FuncAnimation(fig, animate, frames=30, interval=100, repeat=True)
            
            # Save as GIF
            gif_path = "/Users/owner/adk-docs/examples/python/aneurysm_zoom_animation.gif"
            anim.save(gif_path, writer='pillow', fps=10)
            plt.close()
            
            result = {
                "animation_type": "Aneurysm Zoom Animation",
                "file_path": gif_path,
                "frames": 30,
                "fps": 10,
                "status": "success"
            }
            
            print(f"✅ Aneurysm animation created: {gif_path}")
            return result
            
        except Exception as e:
            print(f"❌ Error creating animation: {e}")
            return {"status": "error", "error": str(e)}
    
    async def chapter_5_create_4d_slice_png(self) -> Dict[str, Any]:
        """Execute Chapter 5: Create PNG of 4D scan slice."""
        print("🖼️ Chapter 5: Creating 4D Scan Slice PNG...")
        
        try:
            # Create synthetic 4D volume (time series of 3D volumes)
            time_points = 8
            volume_shape = (32, 32, 32)
            
            # Create base 3D volume
            base_volume = np.random.rand(*volume_shape) * 0.4
            
            # Add anatomical structures
            # Brain outline
            center = (16, 16, 16)
            y, x, z = np.ogrid[:volume_shape[0], :volume_shape[1], :volume_shape[2]]
            brain_mask = np.sqrt((x - center[1])**2 + (y - center[0])**2 + (z - center[2])**2) <= 12
            base_volume[brain_mask] = 0.6
            
            # Create 4D volume with temporal variation
            volume_4d = np.zeros((time_points, *volume_shape))
            for t in range(time_points):
                # Add temporal variation (simulating blood flow)
                temporal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * t / time_points)
                volume_4d[t] = base_volume * temporal_factor
            
            # Select middle time point and slice
            middle_time = time_points // 2
            middle_slice = volume_4d[middle_time, 16, :, :]  # Coronal slice
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(middle_slice, cmap='gray', vmin=0, vmax=1)
            ax.set_title('4D Medical Scan - Coronal Slice (Middle Time Point)')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Intensity')
            
            # Save as PNG
            png_path = "/Users/owner/adk-docs/examples/python/4d_scan_slice.png"
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            result = {
                "image_type": "4D Scan Slice",
                "file_path": png_path,
                "dimensions": f"{volume_shape[1]}x{volume_shape[2]}",
                "time_point": middle_time,
                "status": "success"
            }
            
            print(f"✅ 4D scan slice PNG created: {png_path}")
            return result
            
        except Exception as e:
            print(f"❌ Error creating PNG: {e}")
            return {"status": "error", "error": str(e)}
    
    async def execute_parallel_chapters(self) -> Dict[str, Any]:
        """Execute all chapters in parallel."""
        print("🚀 Executing Parallel Chapters...")
        
        # Create parallel tasks
        tasks = [
            self.chapter_1_reasoning_agent(),
            self.chapter_2_dicom_nifti_agent(),
            self.chapter_3_visualization_agent(),
            self.chapter_4_create_aneurysm_animation(),
            self.chapter_5_create_4d_slice_png()
        ]
        
        # Execute all chapters in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Organize results
        chapter_results = {}
        for i, result in enumerate(results, 1):
            if isinstance(result, Exception):
                chapter_results[f"chapter_{i}"] = {"status": "error", "error": str(result)}
            else:
                chapter_results[f"chapter_{i}"] = result
        
        return chapter_results
    
    async def create_final_agent(self, chapter_results: Dict[str, Any]) -> LlmAgent:
        """Create the final comprehensive agent that combines all capabilities."""
        print("🎯 Creating Final Comprehensive Agent...")
        
        # Extract capabilities from all chapters
        capabilities = []
        for chapter, result in chapter_results.items():
            if result.get("status") == "success":
                if "agent_name" in result:
                    capabilities.append(f"- {result['agent_name']}: {result.get('capabilities', '')}")
                elif "animation_type" in result:
                    capabilities.append(f"- Animation Creation: {result['animation_type']}")
                elif "image_type" in result:
                    capabilities.append(f"- Image Generation: {result['image_type']}")
        
        capabilities_text = "\n".join(capabilities)
        
        final_agent = LlmAgent(
            name="ComprehensiveMedicalImagingAgent",
            model=self.model,
            description="Comprehensive agent combining reasoning, format expertise, and visualization capabilities",
            instruction=f"""
            You are a comprehensive medical imaging AI agent that combines multiple specialized capabilities:
            
            {capabilities_text}
            
            Your integrated capabilities include:
            1. Advanced reasoning for medical image analysis and diagnosis
            2. Expert knowledge of DICOM and NIfTI medical imaging formats
            3. Medical image visualization and animation creation
            4. Aneurysm detection and characterization
            5. 4D medical imaging analysis
            
            You can:
            - Analyze medical images for pathologies like aneurysms
            - Explain differences between medical imaging formats
            - Create visualizations and animations for medical education
            - Provide comprehensive medical imaging consultations
            
            Always maintain medical accuracy and provide evidence-based analysis.
            """,
            tools=[google_search]
        )
        
        print("✅ Final comprehensive agent created!")
        return final_agent
    
    async def run_story(self):
        """Execute the complete medical imaging story."""
        print("=" * 60)
        print("📚 MEDICAL IMAGING STORY: A TALE OF PARALLEL DISCOVERY")
        print("=" * 60)
        
        # Initialize story
        await self.initialize_story()
        
        # Execute parallel chapters
        print("\n🎭 Executing Parallel Chapters...")
        chapter_results = await self.execute_parallel_chapters()
        
        # Create final comprehensive agent
        print("\n🎯 Creating Final Agent...")
        final_agent = await self.create_final_agent(chapter_results)
        
        # Test final agent
        print("\n🧪 Testing Final Agent...")
        runner = InMemoryRunner(
            agent=final_agent,
            app_name=self.app_name,
            session_service=self.session_service
        )
        
        test_query = """
        Demonstrate your comprehensive medical imaging capabilities by:
        1. Explaining how you would analyze a brain scan for aneurysms
        2. Describing when to use DICOM vs NIfTI formats
        3. Outlining your approach to creating medical visualizations
        """
        
        content = types.Content(role="user", parts=[types.Part(text=test_query)])
        
        final_response = ""
        async for event in runner.run_async(
            user_id=self.user_id,
            session_id=self.session_id,
            new_message=content
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        final_response += part.text
        
        # Summary
        print("\n" + "=" * 60)
        print("📖 STORY COMPLETE - SUMMARY")
        print("=" * 60)
        
        print(f"\n🎭 Parallel Chapters Executed: {len(chapter_results)}")
        for chapter, result in chapter_results.items():
            status = "✅" if result.get("status") == "success" else "❌"
            print(f"  {status} {chapter}: {result.get('status', 'unknown')}")
        
        print(f"\n🎯 Final Agent: {final_agent.name}")
        print(f"📝 Capabilities: {final_agent.description}")
        
        print(f"\n📁 Generated Files:")
        for chapter, result in chapter_results.items():
            if "file_path" in result:
                print(f"  📄 {result['file_path']}")
        
        print(f"\n🧠 Final Agent Response Preview:")
        print(final_response[:300] + "..." if len(final_response) > 300 else final_response)
        
        return {
            "story_status": "complete",
            "chapter_results": chapter_results,
            "final_agent": final_agent.name,
            "final_response": final_response
        }


async def main():
    """Main function to run the medical imaging story."""
    story = MedicalImagingStory()
    result = await story.run_story()
    return result


if __name__ == "__main__":
    # Run the story
    asyncio.run(main())
