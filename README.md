# sd_loopback_music_sync_wave

## Overview
#### AUTOMATIC1111 UI extension for creating videos using img2img.  
#### This extension was created based on [Loopback Wave Script](https://github.com/FizzleDorf/Loopback-Wave-for-A1111-Webui)  
#### The major changes are that the wave length can be set one by one in milliseconds and that wildcard can be used.  
#### In addition, I have added various @function.  

## Example
- The following sample is raw output of this extension.  
#### sample 1  




<br>

## Installation  
- Use the Extensions tab of the webui to [Install from URL]  

<br>
<br>

## Basic Usage  
- Go to [txt2img] tab.  
- Generate some image. (I recommend to make with Euler a / 20 steps / cfg 7)  
- Press [Send to img2img] Button  
- Go to [img2img] tab.  
- Lower the [Denoising strength]. (I recommend 0.25)  
- Select [Loopback Music Sync Wave] in Script drop list  
- Copy the following text into the [Wave List (Main)]  
0,wave  
1000,wave  
2000,wave  
3000,wave  
3500,wave  
4000,wave  
4500,wave  
5000,end  
- Copy the following text into the [Extend Prompt (Main)]. The wildcards used below are those provided by default.  
-1::\__lb_vel_slow__  
-1::\__lb_zoom_wave__  
-1::\__lb_prompt_face__  
- Press [Generate]  
- (Default file output location, video encoding settings, etc. are the same as in the original script)  

<br>
<br>

## Advanced Usage  
TODO  
- The list of functions and how to write the wave list are described in [Cheat Sheet].  
- in [Loopback Music Sync Wave] tab, you can automatically generate a wave list. You can also compare prompts and audio waveforms.  

![Cheat Sheet](imgs/cheat_sheet.png "Cheat Sheet")


