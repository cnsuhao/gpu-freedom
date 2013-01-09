unit stereogramsunit;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, configurations;

const MAX_WIDTH = 2048;


type TDepthDataType = Array [0..MAX_WIDTH] of Longint;

procedure makeSameArray(var sameArr : TDepthDataType; size : Longint; xDepthStep : Extended);


implementation

procedure makeSameArray(var sameArr : TDepthDataType; size : Longint; xDepthStep : Extended);
var x : Longint;
    E, ft, depx, xdo, xd : Extended;
begin
  if size>MAX_WIDTH then raise Exception.Create('Image width can not be larger than '+IntToStr(MAX_WIDTH));

  for x:=0 to size-1 do
     sameArr[x] := x;
  E := conf.mEyeDist * conf.mResolution;
  ft := 2 / (conf.mZScale * conf.mMu * E);
  depx := 0;
  xdo  := 0;
  xd := 0;

  // SameArrayType::const_iterator p = pDepth;
  for x:=0 to size-1 do
     begin


     end; // main for loop over x

end;


end.

