unit sphere3dplots;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, texturestructure, conversions, GL, GLU, Graphics,
  ArcBall;

const
   radius = 1;


type  T3DPoint = record
      x,y,z : Real;
end;

type  T3DGrid = Array [0..T_WIDTH] of Array [0..T_HEIGHT] of T3DPoint;
type  P3DGrid = ^T3DGrid;

procedure init3DGrid();
procedure plot3d(vertex : P3DGrid; var colors : TGridColor);
procedure plot3dSphere(var colors : TGridColor);

implementation

var
   sphere3D  : T3DGrid;

procedure init3DGrid();
var i, j : Longint;
    p2 :T3DPoint;
    lat, lon : TFloatType;
begin
 for j := 0 to T_HEIGHT do
     for i := 0 to T_WIDTH do
       begin
         lat := YtoLat(j);       //-90..+90
         lon := XtoLon(i);       //-180..+180

         lat := lat/90 * Pi/2 + Pi/2; //0..180
         lon := lon/180 * Pi + Pi;  //0..360

         // adding half a degree to longitude for triangles
         if (j mod 2 = 0) then
               lon := lon + 1/720 * 2 * Pi;

         p2.x := - radius * sin(lat) * cos(lon);
         p2.z :=   radius * sin(lat) * sin(lon);
         p2.y := - radius * cos(lat);
         sphere3D[i][j] := p2;
       end;

end;


procedure plot3d(vertex : P3DGrid; var colors : TGridColor);
var i, j,
    target_i,
    target_j : Longint;
    p1,p2,p3,p4 : T3DPoint;
    r, g, b,
    correction_x,
    correction_y : TFloatType;

begin
  for j := 0 to T_HEIGHT do
     for i := 0 to T_WIDTH do
       begin
          target_i := i+1;
          target_j := j+1;
          if (target_i>T_WIDTH) then target_i := target_i-T_WIDTH;
          if (target_j>T_HEIGHT) then target_j := target_j-T_HEIGHT;

          p1 := vertex^[i]  [j];
          p2 := vertex^[target_i][j];
          p3 := vertex^[i][target_j];
          p4 := vertex^[target_i][target_j];


          r := Red(colors[i][j])/255;
          g := Green(colors[i][j])/255;
          b := Blue(colors[i][j])/255;

          glBegin(GL_TRIANGLES);
          glColor3f(r,g,b);
          glVertex3f( p1.x, p1.y, p1.z);               // Top Left Of The Triangle (Top)
          glVertex3f( p2.x, p2.y, p2.z);               // Top Right Of The Triangle (Top)
          glVertex3f( p3.x, p3.y, p3.z);               // Bottom Left Of The Triangle (Top)
          glEnd();

          glBegin(GL_TRIANGLES);
          glColor3f(r,g,b);
          glVertex3f( p2.x, p2.y, p2.z);               // Top Right Of The Triangle (Top)
          glVertex3f( p3.x, p3.y, p3.z);               // Bottom Left Of The Triangle (Top)
          glVertex3f( p4.x, p4.y, p4.z);               // Bottom Right Of The Triangle (Top)
          glEnd();
       end;

end;

procedure plot3dSphere(var colors : TGridColor);
begin
 plot3d(@sphere3D, colors);
end;


end.

