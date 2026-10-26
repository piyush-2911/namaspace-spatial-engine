import assert from 'node:assert/strict';
import {createMovementInput} from './dist/movement-input.js';
const win=new EventTarget(), doc=new EventTarget(), viewport={focus(){doc.activeElement=this;}};
const button=new EventTarget();button.dataset={move:'KeyS'};button.setPointerCapture=()=>{};
const input=createMovementInput(win,doc,viewport,[button],()=>true);
const emit=(target,type,props={})=>{const e=new Event(type,{cancelable:true});Object.assign(e,props);target.dispatchEvent(e);};
emit(win,'keydown',{code:'KeyS'});assert.equal(input.has('KeyS'),false,'Unfocused keys must not move camera');
viewport.focus();emit(win,'keydown',{code:'KeyS'});assert.equal(input.has('KeyS'),true);
emit(win,'keyup',{code:'KeyS'});assert.equal(input.has('KeyS'),false,'Release stops backwards movement');
for(const event of ['blur','pagehide']){emit(win,'keydown',{code:'KeyS'});emit(win,event);assert.equal(input.has('KeyS'),false);}
emit(win,'keydown',{code:'KeyS'});emit(doc,'visibilitychange');assert.equal(input.has('KeyS'),false);
emit(win,'keydown',{code:'KeyS'});emit(doc,'focusin');assert.equal(input.has('KeyS'),false);
emit(win,'keydown',{code:'KeyS'});emit(win,'keydown',{code:'Escape'});assert.equal(input.has('KeyS'),false);
emit(button,'pointerdown',{pointerId:1});emit(win,'keydown',{code:'KeyS'});
emit(win,'pointerup',{pointerId:1});assert.equal(input.has('KeyS'),true,'Pointer release must preserve actual keyboard hold');
emit(win,'keyup',{code:'KeyS'});assert.equal(input.has('KeyS'),false);
emit(button,'pointerdown',{pointerId:2});emit(win,'pointercancel',{pointerId:2});assert.equal(input.has('KeyS'),false);
emit(button,'pointerdown',{pointerId:3});emit(button,'lostpointercapture',{pointerId:3});assert.equal(input.has('KeyS'),false);
console.log('Movement regression checks passed: focus, release, blur, visibility, Escape, pointer cancellation, mixed inputs.');
