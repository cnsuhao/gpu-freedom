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

type TDelegates = record
    dels : Array[1..MAX_DELEGATES] of TDelegate;
    size : Longint;
end;

type TParty  = record
    members : Array[1..MAX_DELEGATES] of Longint; // index pointing to delegates
    size    : Longint;
end;

type TParties = record
    par : Array [1..MAX_PARTIES] of TParty;
    size : Longint;
end;

type TParliament = record
    delegates   : TDelegates;
    parties : TParties;
end;

implementation

end.

