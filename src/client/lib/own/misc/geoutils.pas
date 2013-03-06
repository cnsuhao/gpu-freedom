unit geoutils;

interface

uses
 SysUtils;

const
  EARTH_RADIUS    = 6500E3;  // earth radius in meters
  EARTH_PERIMETER = 42000E3; // earth perimeter in meters

// longitudes from -180 to + 180, latitudes from -90 to + 90
function getDistanceOnEarthSphere(lon1, lat1, lon2, lat2 : Extended) : Extended;

implementation


function getDistanceOnEarthSphere(lon1, lat1, lon2, lat2 : Extended) : Extended;
var d,
    dLat,
    dLon : Extended;
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

  //TODO: check if pythagoras holds on sphere // note it does not!!
  dLat := Abs(lat1-lat2)/180*EARTH_PERIMETER/2;
  dLon := Abs(lon1-lon2)/360*EARTH_PERIMETER;
  d := Sqrt(sqr(dLat)+sqr(dLon));
  Result := d;
end;

end.
