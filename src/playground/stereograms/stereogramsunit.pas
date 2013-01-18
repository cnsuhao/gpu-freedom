unit stereogramsunit;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, configurations, ExtCtrls, Graphics;

const MAX_WIDTH = 2048;


type TDepthDataType = Array [0..MAX_WIDTH] of Longint;
type TDepthAssigned = Array [0..MAX_WIDTH] of Boolean;

// preparing inputs
procedure prepareDepthArray(var zimg : TImage; var pDepth : TDepthDataType; y : Longint);
// processing inputs and creating output
procedure makeSameArray(var sameArr, pDepth : TDepthDataType; size : Longint; xDepthStep : Extended);
// processing output
procedure colorImageLineBlackWhite(var sameArr : TDepthDataType; stereoimg : TImage; y : Longint);


procedure printSameArray(var sameArr : TDepthDataType; size : Longint);
procedure checkSameArray(var sameArr : TDepthDataType; size, y : Longint);

implementation

var assigned : TDepthAssigned;

procedure prepareDepthArray(var zimg : TImage; var pDepth : TDepthDataType; y : Longint);
var x : Longint;
begin
  if zimg.Width>MAX_WIDTH then raise Exception.Create('Image width can not be larger than '+IntToStr(MAX_WIDTH));
  for x:=0 to zimg.Width-1 do
     begin
        pDepth[x] := zimg.Picture.Bitmap.Canvas.Pixels[x,y] and 255;
     end;

end;

// xDepthStep = mScaledDepthData.size().width() / mStereogram.width()
procedure makeSameArray(var sameArr, pDepth : TDepthDataType; size : Longint; xDepthStep : Extended);
var x,  xdo, xd, p, ph, s, left, right, l, Zorg  : Longint;
    E, ft, depx, Z : Extended;

    t, ts, zint, zt : Longint;
    visible : Boolean;
begin
  if size>MAX_WIDTH then raise Exception.Create('Image width can not be larger than '+IntToStr(MAX_WIDTH));

  for x:=0 to size-1 do
     sameArr[x] := x;
  E := conf.mEyeDist * conf.mResolution;
  ft := 2 / (conf.mZScale * conf.mMu * E);
  depx := 0;
  xdo  := 0;
  xd := 0;
  p  := 0;

  // SameArrayType::const_iterator p = pDepth;
  for x:=0 to size-1 do
     begin // main for loop over x
       ZOrg  := pDepth[p];
       Z     := ZOrg * conf.mZScale;
       s     := Round(E * (1 - conf.mMu * Z) / (2 - conf.mMu * Z));
       left  := x - Round(s/2);
       right := x + Round(s/2);
       if (left>=0) and (right<size) then
                    begin // inside borders of picture with projection
                          t := 1;
                          repeat  // decides if pixel is visible
                                zt := Zorg + Round( (2 - conf.mMu * Z) * t * ft  );
                                ts := Round (t * xDepthStep);

                                ph := p - ts;
                                visible := (pDepth[ph] < zt);
                                if (visible) then
                                             begin
                                                   ph := p + ts;
                                                   visible := (pDepth[ph] < zt);
                                             end;
                                Inc(t);
                          until (not (visible and (conf.mZscale>zt) ) );

                          if visible then   // finds suitable color for pixel
                                     begin
                                          l := sameArr[left];
                                          while (l<>left) and (l<>right) do
                                          begin
                                                          if (l<right) then
                                                                     begin
                                                                          left := l;
                                                                          l := sameArr[left];
                                                                     end
                                                                    else
                                                                     begin
                                                                          sameArr[left] := right;
                                                                          left := right;
                                                                          l := sameArr[left];
                                                                          right := l;
                                                                     end;


                                          end; // while
                                          sameArr[left] := right;
                                     end;  // if visible, // finds suitable color for pixel

                    end;  // inside borders of picture with projection


     end; // main for loop over x

     depx := depx + xDepthStep;
     xd := Round(depx);
     p := p + xd - xdo;
     xdo := xd;

end;


procedure colorImageLineBlackWhite(var sameArr : TDepthDataType; stereoimg : TImage; y : Longint);
var x, k : Longint;
    c : TColor;
begin
   // init assigned array
   for x:=0 to stereoimg.Width-1 do assigned[x] := false;

   c := clWhite;
   for x:=0 to stereoimg.Width-1 do
        begin
           if assigned[x] then continue;

           stereoimg.Picture.Bitmap.Canvas.Pixels[x,y] := c;
           assigned[x] := true;

           k:=x;
           while (k<>sameArr[k]) and (k<stereoimg.Width) do
             begin
                k:=sameArr[k];
                stereoimg.Picture.Bitmap.Canvas.Pixels[k,y] := c;
                assigned[k] := true;
             end;


           if (c=clWhite) then c:=clBlack else c:=clWhite;
        end;

end;

procedure log(str, filename : AnsiString);
var F : Textfile;
begin
  if not FileExists(filename) then
     begin
       AssignFile(F, filename);
       Rewrite(F);
       CloseFile(F);
     end;

  AssignFile(F, filename);
  Append(F);
  WriteLn(F, str);
  CloseFile(F);
end;

procedure printSameArray(var sameArr : TDepthDataType; size : Longint);
var output : AnsiString;
    i : Longint;
begin
    output := '';
    for i:=0 to size-1 do
       output := output + IntToStr(sameArr[i])+';';
    log(output, 'samearr.txt');
end;

procedure checkSameArray(var sameArr : TDepthDataType; size, y : Longint);
var
    x : Longint;
begin
  for x:=0 to size-1 do
       begin
          if sameArr[x]<x then
               begin
                 log('Consistency violated: error at line '+IntToStr(y)+', position '+IntToStr(x), 'error.txt');
                 Exit;
               end;
       end;
end;

end.

