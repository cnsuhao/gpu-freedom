unit conversion;
{(c) 2010 HB9TVM    Source code is under GPL}

interface
uses Sysutils;

  function LonToX(lon : Extended) : Longint;
  function LatToY(lat : Extended) : Longint;

  function XtoLon(x : Longint) : Extended;
  function YtoLat(y : Longint) : Extended;

  function KtoC(k : Extended) : Extended;
  function CtoK(c : Extended) : Extended;

  procedure InitConversion(nonlinear : Boolean);
  function isNonLinearConversion : Boolean;

implementation

var latitudeToY : Array [0..90] of Longint;
    YtoLatitude : Array [0..90] of Extended;
    nonlinear_ : Boolean;

function isNonLinearConversion : Boolean;
begin
  Result := nonlinear_;
end;


function LonToX(lon : Extended) : Longint;
begin
  if (lon<-180) or (lon > 180) then
     raise Exception.create('Longitude has to be between -180 (W) and +180 (E) but was '+FloatToStr(lon));
  if (lon=180) then Result := 0     // 180 deg E = 180 deg W
  else
    Result := Trunc(lon + 180);
end;

function XtoLon(x : Longint) : Extended;
begin
  if (x<0) or (x > 359) then
     raise Exception.create('x on array has to be between 0 and 359 but was '+IntToStr(x));

  Result := x - 180;
end;

function LatToY_lin(lat : Extended) : Longint;
begin
   Result := Trunc(90 - lat);
end;

function LatToY_nonlin(lat : Extended) : Longint;
var
 trn  : Longint;
 prt  : Extended;
begin
  trn := Trunc(lat);
  prt := Abs(Frac(lat));

  if Abs(lat)<0.5 then
    Result := 90
  else
  if (lat<0) then
    begin
           if (prt=0.5) then
             Result := Trunc( (  (180-LatitudeToY[Abs(trn)])  +   (180-LatitudeToY[Abs(trn)+1]) )/2 )
           else
           if (prt<0.5) then
                      Result := 180-LatitudeToY[Abs(trn)]
           else
                      Result := 180-LatitudeToY[Abs(trn)+1];
    end
  else
    begin
       // lat > 0;
       if (prt=0.5) then
           Result := Trunc( (LatitudeToY[trn+1]+LatitudeToY[trn])/2 )
       else
       if (prt<0.5) then
           Result := LatitudeToY[trn]
       else
           Result := LatitudeToY[trn+1];
   end;

end;


function YtoLat_lin(y : Longint) : Extended;
begin
  Result := 90 - y;
end;

function YtoLat_nonlin(y : Longint) : Extended;
begin
  if (y<=90) then
    Result := YtoLatitude[y]
  else
    Result := -YtoLatitude[180-y];
end;

function LatToY(lat : Extended) : Longint;
begin
   if (lat<-90) or (lat > 90) then
     raise Exception.create('Latitude has to be between -90 (S) and +90 (N) but was '+FloatToStr(lat));

  if nonlinear_ then
    Result := LatToY_nonlin(lat)
  else
    Result := LatToY_lin(lat);

  if Result = 180 then Result := 179; // Latitude -90 maps to 179
end;

function YtoLat(y : Longint) : Extended;
begin
  if (y<0) or (y > 180) then
     raise Exception.create('y on array has to be between 0 and 180 but was '+IntToStr(y));

  if nonlinear_ then
    Result := YToLat_nonlin(y)
  else
    Result := YToLat_lin(y);
end;


function KtoC(k : Extended) : Extended;
begin
 if (k<0) then
   raise Exception.create('Temperature in Kelvin can not be negative but was '+FloatToStr(k));

 Result := k - 273.16;
end;

function CtoK(c : Extended) : Extended;
begin
 if (c<-273.16) then
   raise Exception.create('Temperature in Celsius has to be higher or equal than absolute zero (-273.16) but was '+FloatToStr(c));
 Result := c + 273.16;
end;

procedure InitConversion(nonlinear : Boolean);
var j : Longint;
begin
  nonlinear_ := nonlinear;

     // init nonlinear merges and splits tables
     YtoLatitude[ 0] := 90;
      LatitudeToY[90] := 0;
      LatitudeToY[89] := 0;

     YtoLatitude[ 1] := 88;
      LatitudeToY[88] := 1;
      LatitudeToY[87] := 1;

     YtoLatitude[ 2] := 86;
      LatitudeToY[86] := 2;
      LatitudeToY[85] := 2;

     YtoLatitude[ 3] := 84;
      LatitudeToY[84] := 3;
      LatitudeToY[83] := 3;

     YtoLatitude[ 4] := 82;
      LatitudeToY[82] := 4;
      LatitudeToY[81] := 4;

     // bijection
     for j:=5 to 79 do
      begin
       YtoLatitude[ j] := 85-j; LatitudeToY[85-j] := j;
      end;

     // splits
     YtoLatitude[80] := 5;    LatitudeToY[5] := 80;
     YtoLatitude[81] := 4.5;

     YtoLatitude[82] := 4;    LatitudeToY[4] := 82;
     YtoLatitude[83] := 3.5;

     YtoLatitude[84] := 3;    LatitudeToY[3] := 84;
     YtoLatitude[85] := 2.5;

     YtoLatitude[86] := 2;    LatitudeToY[2] := 86;
     YtoLatitude[87] := 1.5;

     YtoLatitude[88] := 1;    LatitudeToY[1] := 88;
     YtoLatitude[89] := 0.5;

     YtoLatitude[90] := 0;    LatitudeToY[0] := 90;
end;


end.
