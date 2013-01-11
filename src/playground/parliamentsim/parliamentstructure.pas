unit parliamentstructure;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils;

const MAX_DELEGATES = 300;
      MAX_PARTIES = 5;

type TDelegate = record
    personalinterest,
    collectiveinterest : Extended; // values between -1..1
end;

type TDelegates = Array[1..MAX_DELEGATES] of TDelegate;

type TParty  = record
    members : TDelegates;
    size    : Longint;
end;

type TParties = Array [1..MAX_PARTIES] of TParty;

type TParliament = record
    nbdelegates : Longint;
    nbparties : Longint;
    parties : TParties;
end;

implementation

end.

