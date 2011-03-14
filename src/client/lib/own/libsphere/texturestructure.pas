unit texturestructure;

interface

uses colors;

const
   T_WIDTH  = 359;
   T_HEIGHT = 179;

type  TFloatType   = Extended;
type  TGridFloat   = Array [0..T_WIDTH] of Array [0..T_HEIGHT] of TFloatType;
type  TGridBoolean = Array [0..T_WIDTH] of Array [0..T_HEIGHT] of Boolean;
type  TGridColor   = Array [0..T_WIDTH] of Array [0..T_HEIGHT] of TColor;
type  PGridColor   = ^TGridColor;

type TSphereTexturer = class(TObject)
 public
    procedure fillTexture(var colors : TGridColor); virtual; abstract;
end;

implementation


end.
