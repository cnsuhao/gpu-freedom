unit jobstatstables;

interface

uses sqlite3ds, db, coretables, SysUtils;

type TDbJobStatsRow = record
   id         : Longint;
   jobdefinitionid  : String;
   job        : AnsiString;
   jobtype    : String;
   requireack : Boolean;
   transmitted,
   received,
   acknowledged : Longint;
   server_id    : Longint;
   create_dt    : TDateTime;
end;

implementation

end;
