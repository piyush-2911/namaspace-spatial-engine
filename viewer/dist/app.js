import * as THREE from 'three';
import {Viewer,SceneFormat} from './vendor/gaussian-splats-3d.module.js';
import {createMovementInput} from './movement-input.js';
const $=id=>document.getElementById(id), viewport=$('viewport');
const keys=createMovementInput(window,document,viewport,document.querySelectorAll('[data-move]'),()=>ready && !$('help-dialog').open);
let viewer,camera,config,ready=false,activeView=0,drag=null,lastTime=0,frames=0,fpsStart=0,sortDirty=false,lastSortTime=0;
const worldUp=new THREE.Vector3(), forward=new THREE.Vector3(),right=new THREE.Vector3(),movement=new THREE.Vector3();
const turn=new THREE.Quaternion();
function selectView(index){
  if(!camera)return;activeView=index;
  const v=config.views[index], p=new THREE.Vector3(...v.position), target=new THREE.Vector3(...v.target);
  camera.position.copy(p);camera.up.copy(worldUp);camera.lookAt(target);camera.updateMatrixWorld(true);keys.clear();if(ready)viewer.runSplatSort(true,true);
  document.querySelectorAll('#views button').forEach((b,i)=>b.classList.toggle('active',i===index));
}
function look(dx,dy){
  if(!ready)return;
  sortDirty=true;turn.setFromAxisAngle(worldUp,-dx*.0025);camera.quaternion.premultiply(turn);
  right.set(1,0,0).applyQuaternion(camera.quaternion);
  const old=camera.quaternion.clone();
  turn.setFromAxisAngle(right,-dy*.0025);camera.quaternion.premultiply(turn);
  camera.getWorldDirection(forward);
  if(Math.abs(forward.dot(worldUp))>.985)camera.quaternion.copy(old);
}
viewport.tabIndex=0;
viewport.addEventListener('pointerdown',e=>{if(!ready)return;viewport.focus({preventScroll:true});drag={x:e.clientX,y:e.clientY,id:e.pointerId};viewport.setPointerCapture(e.pointerId);});
viewport.addEventListener('pointermove',e=>{if(!drag||drag.id!==e.pointerId)return;look(e.clientX-drag.x,e.clientY-drag.y);drag.x=e.clientX;drag.y=e.clientY;});
for(const event of ['pointerup','pointercancel','lostpointercapture'])viewport.addEventListener(event,()=>drag=null);
window.addEventListener('keydown',e=>{if(e.code==='KeyR' && document.activeElement===viewport && !$('help-dialog').open)selectView(0);if(e.code==='Escape')drag=null;});
window.addEventListener('blur',()=>drag=null);
$('reset').onclick=()=>selectView(0);$('help').onclick=()=>{keys.clear();$('help-dialog').showModal();};$('close-help').onclick=()=>$('help-dialog').close();
$('fullscreen').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{$('hint').textContent='Fullscreen is unavailable in this browser.';}};
$('retry').onclick=()=>location.reload();
const selectedScene=new URLSearchParams(location.search).get('scene')==='living'?'living':'meeting';
document.querySelectorAll('[data-scene]').forEach(link=>{
  if(link.dataset.scene===selectedScene)link.setAttribute('aria-current','page');
  link.addEventListener('click',()=>keys.clear());
});
function tick(time){
  requestAnimationFrame(tick);
  const dt=Math.min((time-lastTime)/1000,.05);lastTime=time;
  if(!ready)return;
  camera.getWorldDirection(forward);right.set(1,0,0).applyQuaternion(camera.quaternion);movement.set(0,0,0);
  if(keys.has('KeyW')||keys.has('ArrowUp'))movement.add(forward);
  if(keys.has('KeyS')||keys.has('ArrowDown'))movement.sub(forward);
  if(keys.has('KeyD')||keys.has('ArrowRight'))movement.add(right);
  if(keys.has('KeyA')||keys.has('ArrowLeft'))movement.sub(right);
  if(keys.has('KeyE'))movement.add(worldUp);if(keys.has('KeyQ'))movement.sub(worldUp);
  if(movement.lengthSq()){sortDirty=true;camera.position.addScaledVector(movement.normalize(),dt*Number($('speed').value)*config.movementScale*(keys.has('ShiftLeft')||keys.has('ShiftRight')?3:1));}
  camera.updateMatrixWorld();if(sortDirty && time-lastSortTime>100 && !viewer.sortRunning){viewer.runSplatSort(true,true);sortDirty=false;lastSortTime=time;}viewer.update();viewer.render();frames++;
  if(time-fpsStart>1000){$('fps').textContent=Math.round(frames*1000/(time-fpsStart))+' FPS';frames=0;fpsStart=time;}
}
async function init(){
  try{
    const scenePath=selectedScene==='living'?'./living/capture-3/scene.json?v=views-3':'./scene.json';
    const response=await fetch(scenePath);if(!response.ok)throw Error('Scene configuration could not be loaded.');config=await response.json();
    if(config.title)document.querySelector('h1').textContent=config.title;
    if(config.captureLabel)document.querySelector('.tags span').textContent=config.captureLabel;
    if(config.note)document.querySelector('.scene-card .note').textContent=config.note;
    worldUp.fromArray(config.up).normalize();
    config.views.forEach((v,i)=>{const b=document.createElement('button');b.disabled=true;b.innerHTML=`<span>${v.label}</span><span class="num">0${i+1} ↗</span>`;b.onclick=()=>selectView(i);$('views').appendChild(b);});
    camera=new THREE.PerspectiveCamera(config.fov,innerWidth/innerHeight,.01,1000);
    selectView(0);
    const renderer=new THREE.WebGLRenderer({antialias:false,alpha:false,powerPreference:'high-performance'});
    renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setSize(innerWidth,innerHeight);viewport.appendChild(renderer.domElement);
    renderer.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();ready=false;$('load-title').textContent='Graphics connection lost';$('load-text').textContent='Reload the viewer to reconnect to your graphics device.';$('retry').hidden=false;$('loading').hidden=false;});
    viewer=new Viewer({rootElement:viewport,renderer,camera,selfDrivenMode:false,useBuiltInControls:false,sharedMemoryForWorkers:false,gpuAcceleratedSort:false,integerBasedSort:true,sphericalHarmonicsDegree:0});
    window.addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
    let received=0;const parts=[];
    for(const path of config.parts){
      const res=await fetch(path);if(!res.ok)throw Error('A model file could not be downloaded.');
      if(res.body){const reader=res.body.getReader();const chunks=[];while(true){const {done,value}=await reader.read();if(done)break;chunks.push(value);received+=value.byteLength;$('progress').value=90*received/config.bytes;$('load-text').textContent=`${(received/1e6).toFixed(1)} / ${(config.bytes/1e6).toFixed(1)} MB`; }parts.push(new Blob(chunks));}
      else {const data=await res.arrayBuffer();received+=data.byteLength;parts.push(data);}
    }
    if(received!==config.bytes)throw Error('The model download was incomplete. Please try again.');
    $('load-title').textContent='Building your view';$('load-text').textContent='Preparing the scene for interactive exploration…';$('progress').value=94;
    const url=URL.createObjectURL(new Blob(parts));
    try{await viewer.addSplatScene(url,{format:SceneFormat.Splat,showLoadingUI:false,progressiveLoad:false,splatAlphaRemovalThreshold:1});}finally{URL.revokeObjectURL(url);}
    ready=true;$('reset').disabled=false;document.querySelectorAll('#views button').forEach(b=>b.disabled=false);$('loading').hidden=true;selectView(0);
    document.body.dataset.sceneReady='true';
  }catch(error){console.error(error);$('load-title').textContent='The room couldn’t open';$('load-text').textContent=error.message+' Try Chrome or Edge with graphics acceleration enabled.';$('retry').hidden=false;$('progress').hidden=true;}
}
requestAnimationFrame(tick);init();





