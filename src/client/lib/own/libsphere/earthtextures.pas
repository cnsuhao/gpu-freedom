unit earthtextures;

interface

uses SysUtils, Math, ExtCtrls, texturestructure, utils, colors, conversions;


type TEarthTexturer = class(TSphereTexturer)
  public
   constructor Create;
   destructor Destroy;
   function getTexture() : PGridColor; virtual;
private
   appPath_    : String;
   colors_     : TGridColor;

   imgEarth_   : TImage;

   procedure paintTexture;
end;

implementation

constructor TEarthTexturer.Create;
begin
 appPath_ := ExtractFilePath(ParamStr(0));

 imgEarth_ := TImage.Create(nil);
 imgEarth_.Picture.LoadFromFile(appPath_+'\data\textures\solarsystem\earth.bmp');

 paintTexture;
end;

destructor TEarthTexturer.Destroy;
begin
 imgEarth_.free;
 inherited Destroy;
end;

function TEarthTexturer.getTexture() : PGridColor;
begin
  Result := @colors_;
end;


procedure TEarthTexturer.paintTexture();
var i, j : Longint;
begin
  for j:=0 to T_HEIGHT do
    for i:=0 to T_WIDTH do
      begin
            colors_[i][j] := imgEarth_.Picture.Bitmap.Canvas.Pixels[i*2,j*2];
      end;
end;

end.
