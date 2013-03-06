unit geoutils;

interface

uses
 SysUtils, Math;

const
  EARTH_RADIUS    = 6371E3;  // earth radius in meters

// longitudes from -180 to + 180, latitudes from -90 to + 90
function getDistanceOnEarthSphere(lat1, lon1, lat2, lon2 : Extended) : Extended;
procedure testDistance;

implementation

// input angle in degrees, output angle in radians
function radians(angle : Extended) : Extended;
begin
  Result := angle/360*2*Pi;
end;

function degrees(angle : Extended) : Extended;
begin
  Result := angle/(2*Pi)*360;
end;

function getDistanceOnEarthSphere(lat1, lon1, lat2, lon2 : Extended) : Extended;
var d,
    dLat,
    dLon,
    p1, p2, p3 : Extended;
begin
  if (lon1<-180) or (lon1>180) then raise Exception.Create('geoutils.getDistanceOnEarthSphere: lon1 has to be between -180 and 180');
  if (lon2<-180) or (lon2>180) then raise Exception.Create('geoutils.getDistanceOnEarthSphere: lon2 has to be between -180 and 180');
  if (lat1< -90) or (lat1> 90) then raise Exception.Create('geoutils.getDistanceOnEarthSphere: lat1 has to be between -90 (south pole) and 90 (north pole)');
  if (lat2< -90) or (lat2> 90) then raise Exception.Create('geoutils.getDistanceOnEarthSphere: lat2 has to be between -90 (south pole) and 90 (north pole)');

  // normalizing angles to 0..360
  lon1 := lon1+180;
  lon2 := lon2+180;
  // normalizing angles to 0..180
  lat1 := lat1+90;
  lat2 := lat2+90;

  // correct formula on http://en.wikipedia.org/wiki/Great-circle_distance
  // approximation on http://jsp.vs19.net/lr/sphere-distance.php by  C.J. Spaans
  p1 := cos(radians(lon1-lon2));
  p2 := cos(radians(lat1-lat2));
  p3 := cos(radians(lat1+lat2));

  Result := arccos(((p1*(p2+p3))+(p2-p3))/2)  * EARTH_RADIUS;
end;

procedure testDistance;
begin
  WriteLn(FloatToStr(getDistanceOnEarthSphere(51.9897, 4.3759,  52.0103, 4.3661)));
  WriteLn('Result should be around 2390 m');
end;

end.
