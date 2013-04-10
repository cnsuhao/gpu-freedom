unit jobapis;

interface

uses coreapis, dbtablemanagers;


type TJobTransmissionDetails = record
     workunitjob : String;
     workunitresult : String;
     nbrequests  : Longint;
     tagwujob,
     tagwuresult : Boolean;
end;

TGPUJobAPI = record
    jobdefinitionid : String;
    job             : AnsiString;
    jobtype         : String;
    requireack,
    islocal         : Boolean;

    trandetails     : TJobTransmissionDetails;
end;


type TJobAPI = class(TCoreAPI)
  public
    constructor Create(var tableman : TDbTableManager);

end;


implementation

constructor TJobAPI.Create(var tableman : TDbTableManager);
begin
  inherited Create(tableman);
end;

end.
