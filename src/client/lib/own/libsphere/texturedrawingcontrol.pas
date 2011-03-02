unit textureDrawingControl;

{$mode objfpc}{$H+}
interface

uses
  Classes, SysUtils, Controls, Graphics, LCLType, Dialogs,
  colors, texturestructure;


type
  TTextureDrawingControl = class(TCustomControl)
  public
    constructor Create(obj : TComponent; plotmode : Longint; size : Longint; colors : PGridColor);
    procedure Paint; override;
    procedure setPlotMode(plotmode : Longint);
    function  getPlotMode : Longint;
    function  getColors : PGridColor;
    procedure setColors(colors : PGridColor);

    procedure MouseDown(Button: TMouseButton; Shift: TShiftState; X, Y: Integer); override;


  private
    _colors : TGridColor;

    _size : Longint; //1,2,3,4
    _plotMode : Longint;
  end;

  type PEarthDrawingControl = ^TEarthDrawingControl;

implementation


constructor TTextureDrawingControl.Create(obj : TComponent; plotmode : Longint; size : Longint; colors : PGridColor);
begin
  inherited Create(obj);
  _size := size;
  _plotMode := plotmode;
  _clima := clima;

  Height := _size*180+1;
  Width := _size*360+1;
  setColors(colors);
end;

procedure TTextureDrawingControl.setPlotMode(plotmode : Longint);
begin
 _plotMode := plotMode;
end;

function TTextureDrawingControl.getPlotMode : Longint;
begin
 Result := _plotMode;
end;

function TTextureDrawingControl.getColors : PGridColor;
begin
  Result := @_colors;
end;

procedure TTextureDrawingControl.setColors(colors : PGridColor);
var i, j : Longint;
begin
  for j := 0 to 179 do
     for i := 0 to 359 do
        _colors[i][j] := colors[i][j]^;
end;


procedure TTextureDrawingControl.Paint;
var
  i, j: Integer;
  Bitmap: TBitmap;
begin
  Bitmap := TBitmap.Create;
  try
    Bitmap.Height := Height;
    Bitmap.Width := Width;

    for j := 0 to 179 do
     for i := 0 to 359 do
       begin
         Bitmap.Canvas.Pen.Color := colors[i][j];
         Bitmap.Canvas.Brush.Color := colors[i][j];
         if (_size>1) then
           Bitmap.Canvas.Rectangle(i*_size ,j*_size, i*_size+_size, j*_size+_size)
         else
          begin
           Bitmap.Canvas.MoveTo(i, j);
           Bitmap.Canvas.LineTo(i+1, j+1);
          end;
       end;

    Canvas.Draw(0, 0, Bitmap);
  finally
    Bitmap.Free;
  end;

  inherited Paint;
end;

procedure TTextureDrawingControl.MouseDown(Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
begin
  inherited MouseDown(Button, Shift,X, Y);
end;

end.

