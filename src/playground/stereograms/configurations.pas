unit configurations;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils;

const INCH_IN_CM = 2.54;

type TConfiguration = record
    mEyeDist,
    mResolution,
    mZScale,
    mMu       : Extended;
end;

var conf : TConfiguration;

procedure initConfiguration();
procedure loadConfiguration(E, monitorWidthPx, monitorWidthCm, mu : Extended);

implementation

procedure initConfiguration();
begin
  conf.mEyeDist := 2.56;   // (6.5cm in inches)
  conf.mResolution := 85; // pixels per inch
  conf.mZScale := 1/256;
  conf.mMu := 1/3;
end;


procedure loadConfiguration(E, monitorWidthPx, monitorWidthCm, mu : Extended);
begin
  initConfiguration();

  conf.mEyeDist := E / INCH_IN_CM;
  // what to do with twoD??
  conf.mResolution := monitorWidthPx/monitorWidthCm * INCH_IN_CM;

  conf.mMu := mu;
end;


end.

