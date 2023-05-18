# sd_loopback_music_sync_wave

## Overview
#### AUTOMATIC1111 UI extension for creating videos using img2img.  
#### This extension was created based on [Loopback Wave Script](https://github.com/FizzleDorf/Loopback-Wave-for-A1111-Webui)  
#### The major changes are that the wave length can be set one by one in milliseconds and that wildcard can be used.  
#### In addition, I have added various @function.  

## Example
- The following sample is raw output of this extension.(The file was too large, so I compressed it.)  
#### sample 1  
```
Extend Prompts:
1::#zoom(@wave_amplitude(0.8,1.6))
2::$random_xy(3434, 0.2,0.2, 3000)
3::#rot_y(@wave_amplitude(60,0))

Sub Extend Prompts:
1::@@bpm139@1700[#slide_x(1, @random(1.5,2), @random(0.2,0.8))]
3::@@bpm139@1700[#slide_y(1, @random(1.5,2), @random(0.2,0.8))]
5::@@bpm139@1700[#slide_x(1, @random(-1.5,-2), @random(0.2,0.8))]
7::$random_xy(99999, 1,1, 3000),$random_slide_x(99999,1,0.5,0.5,0.5,3000),$random_slide_y(99999,1,0.5,0.5,0.5,3000)
-1::(__expression__:@wave_shape(1.0,0))
```
<div><video controls src="https://user-images.githubusercontent.com/118420657/235941290-2708913c-16c0-4f3a-82db-7ab729ea7ea8.mp4" muted="false"></video></div>
<br>

#### sample 2  
- Loopback mode(default) + Controlnet(open pose)  
- All the parts except for the main prompt are taken from sample 1.  
<div><video controls src="https://user-images.githubusercontent.com/118420657/236166342-6b2457a0-7d4e-4dd5-9964-0d3293c8dc9c.mp4" muted="false"></video></div>
<br>

#### sample 3  
- img2img mode + Controlnet(open pose)  
- The setting is exactly the same as sample2 except mode is changed to img2img.  
<div><video controls src="https://user-images.githubusercontent.com/118420657/236167376-e914c244-c5fc-453d-9b20-32a80c53b9af.mp4" muted="false"></video></div>
<br>

#### sample 4  
- SAM + Controlnet(open pose)  
- Using [SAM](https://github.com/continue-revolution/sd-webui-segment-anything) to dynamically generate mask from text, then inpaint with mask
```
4-15::(__hair-female__: 1.2)
4-15::$inpaint("hair")
20-31::(__clothing__: 1.0)
20-31::$inpaint("clothing")
32::(spiderman:1.5)
33::(wonder woman:1.5)
34::(storm:1.5)
35::(harley_quinn:1.5)
32-35::$inpaint("face")
```
<div><video controls src="https://user-images.githubusercontent.com/118420657/236435879-a07eca27-a443-4888-9f66-8f8027167480.mp4" muted="false"></video></div>

<br>

#### sample 5  
- Loopback mode(with low denoising strength) + optical flow + Controlnet(open pose + normalbae)
- openpose / weight 1.0 / "My prompt is more important"
- normalbae / weight 0.5 / "My prompt is more important"
- (controlnet ip2p also seemed to work well with loopback)
<div><video controls src="https://github.com/s9roll7/sd_loopback_music_sync_wave/assets/118420657/76335d01-3397-4561-b2bf-ac224f76edd7" muted="false"></video></div>

<br>

#### sample 6  
- openpose_full / weight 1.0 / "My prompt is more important"
- reference_adain / weight 1.0 / "Balanced" / threshold_a 0.5
- softedge_pidisafe / weight 0.7 / "My prompt is more important"
- Fps 8 / Interpolation Multiplier 3
```
0:: cyberpunk city
1-100::(__location__: 1.0)
1-100::(__clothing__: 1.0)
-1::(__expression__:@wave_shape(1.0,0))
```
<div><video controls src="https://github.com/s9roll7/sd_loopback_music_sync_wave/assets/118420657/61986379-fbcd-4767-a5f9-143810e57352" muted="false"></video></div>

<br>

## Installation  
- Use the Extensions tab of the webui to [Install from URL]  

<br>
<br>

## Basic Usage 1 (For make a loopback video without a source video)  
- Go to [txt2img] tab.  
- Generate some image. (I recommend to make with Euler a / 20 steps / cfg 7)  
- Press [Send to img2img] Button  
- Go to [img2img] tab.  
- Lower the [Denoising strength]. (I recommend 0.25)  
- Select [Loopback Music Sync Wave] in Script drop list  
- Copy the following text into the [Wave List (Main)]  
```
0,wave  
1000,wave  
2000,wave  
3000,wave  
3500,wave  
4000,wave  
4500,wave  
5000,end  
```
- Copy the following text into the [Extend Prompt (Main)]. The wildcards used below are those provided by default.  
```
-1::__lb_vel_slow__  
-1::__lb_zoom_wave__  
-1::__lb_prompt_face__  
```
- Press [Generate]  
- (Default file output location, video encoding settings, etc. are the same as in the original script)  

<br>
<br>

## Basic Usage 2 (For change the character or style of the source video using the loopback technique)  
- First, prepare your source video (SD or HD resolution, approx. 10 seconds in length)
- Configure controlnet settings.
- Extract the first frame of the source video, then Create the first input image with img2img based on it.
- Set [Project Directory(optional)] / [Video File Path(optional)] / [Mode Settings] / [Optical Flow Setting]  
(I recommend [Frames per second] = 8 and [interpolation multiplier] = 3)
- Press [Generate]  
See [Here](https://github.com/s9roll7/sd_loopback_music_sync_wave/wiki/loopback---controlnet) and [Here](https://github.com/s9roll7/sd_loopback_music_sync_wave/wiki/loopback---controlnet---optical-flow) for more information.

<br>
<br>

## Advanced Usage  

### How to generate video synchronized to music  
See [Here](https://github.com/s9roll7/sd_loopback_music_sync_wave/wiki/How-to-generate-video-synchronized-to-music)
<br>


### How to replace the initial image in the middle of the process  
See [Here](https://github.com/s9roll7/sd_loopback_music_sync_wave/wiki/How-to-replace-the-initial-image-in-the-middle-of-the-process)
<br>

### @#$function list  
- The list of functions and how to write the wave list are described in [Cheat Sheet].  
![Cheat Sheet](imgs/cheat_sheet.png "Cheat Sheet")  
<br>

### loopback + controlnet (sample 2)  
See [Here](https://github.com/s9roll7/sd_loopback_music_sync_wave/wiki/loopback---controlnet)
<br>

### loopback + controlnet + optical flow (sample 5)
See [Here](https://github.com/s9roll7/sd_loopback_music_sync_wave/wiki/loopback---controlnet---optical-flow)
<br>

### How to generate mask from text (sample 4)  
- You need [SAM Extension](https://github.com/continue-revolution/sd-webui-segment-anything)  
- It is necessary to be able to use SAM and GroundingDINO together  
- Refer to sample 4 for $inpaint function usage.  

<br>

### img2img mode (sample 3)  
- As an added bonus, img2img mode is implemented. You can switch between loopback and img2img in [Mode Settings].
- ~~The original frame required for img2img must be generated by the following procedure. (If you want to use controlnet in loopback mode as in sample2, you should also generate frames using this procedure.)~~  
<br>


### Other Tips  
- If you specify the *-inputs.txt that is created at the same time as creating the video in [Load inputs txt Path], you can create the video again with the same input as last time  
- If you want to reuse only some of the inputs, open *-prompt.txt and copy only where you need it
- If you want to add more wildcards yourself,
Put them in [extensions/sd_loopback_music_sync_wave/wildcards]. If you are too lazy to make your own, you can pick them up at civitai.  
- If you want to make a video only for the first 5 seconds in the test, temporarily add "5000, end" to the wave list entered in [Wave List (Main)]  
- The unit of velocity for function parameters is the number of screens per second. For example, a speed of 1.0 in the x-axis direction means that if the screen moves at the same speed for one second, it will scroll one horizontal screen. In the case of rotation speed, it is the degree per second.  
- There are three ways to increase the resolution.  
  1. Do img2img with a higher resolution from the beginning.  
  2. Use [Upscale Settings].  
  3. Upscale the generated video using an external tool.  
1 is probably the best way to get the best results, but it also takes the most processing time.
- There are three ways to increase fps for smooth animation  
  1. Put a larger value in [Frames per second]  
  2. Use interpolation with Optical Flow(When using the source video to generate)
  3. Interpolate with an external tool  
1 is very effective, but the processing time will be proportionally longer.  
