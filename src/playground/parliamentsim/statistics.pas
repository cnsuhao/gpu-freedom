unit statistics;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, parliamentstructure;

  procedure collectStatistics(var parliament : TParliament; var laws : TLaws; var stats : TSimStats; round : Longint);

implementation

function calculateApprovalrate(var parliament : TParliament; var laws : TLaws; round : Longint) : Extended;
var i : Longint;
    approved : Longint;
begin
  approved := 0;
  for i:=1 to laws.size do
    if laws.laws[i].approved then Inc(approved);

  laws.approved:=approved;

  Result := laws.approved/laws.size;
end;

function calculateTotalBenefit(var parliament : TParliament; var laws : TLaws; round : Longint) : Extended;
var i, count       :  Longint;
    benefit        : Extended;
begin
  //for i:=1 to laws.size do
  //  if laws.laws[i].approved;
end;

procedure collectStatistics(var parliament : TParliament; var laws : TLaws; var stats : TSimStats; round : Longint);
begin
  stats.legislatures[round].nbdelegates:=parliament.delegates.size;
  stats.legislatures[round].nbindipendents:=parliament.indipendents;
  stats.legislatures[round].nbparties:=parliament.parties.size;

  stats.legislatures[round].approvalrate:=calculateApprovalrate(parliament, laws, round);
  stats.legislatures[round].totalbenefit:=calculateTotalBenefit(parliament, laws, round);
end;

end.

