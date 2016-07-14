-- phpMyAdmin SQL Dump
-- version 4.5.1
-- http://www.phpmyadmin.net
--
-- Host: 127.0.0.1
-- Creato il: Giu 07, 2016 alle 17:19
-- Versione del server: 10.1.10-MariaDB
-- Versione PHP: 5.6.15

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `meteo`
--

-- --------------------------------------------------------

--
-- Struttura della tabella `tbmeteo_spot`
--

CREATE TABLE `tbmeteo_spot` (
  `countrycode` varchar(5) NOT NULL,
  `referencedate` date NOT NULL,
  `hour` varchar(3) NOT NULL,
  `minute` varchar(3) NOT NULL,
  `id_station` varchar(16) NOT NULL,
  `temperature` double DEFAULT NULL,
  `sun_duration` double DEFAULT NULL,
  `rain` double DEFAULT NULL,
  `wind_direction` double DEFAULT NULL,
  `wind_speed` double DEFAULT NULL,
  `wind_max` double DEFAULT NULL,
  `relative_humidity` double DEFAULT NULL,
  `pressure_QNH` double DEFAULT NULL,
  `pressure_QFE` double DEFAULT NULL,
  `pressure_QFF` double DEFAULT NULL,
  `insertdate` datetime DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Struttura della tabella `tbparameter_desc`
--

CREATE TABLE `tbparameter_desc` (
  `countrycode` varchar(5) NOT NULL,
  `id_parameter` varchar(16) NOT NULL,
  `fieldname` varchar(32) NOT NULL,
  `unit` varchar(10) NOT NULL,
  `description` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dump dei dati per la tabella `tbparameter_desc`
--

INSERT INTO `tbparameter_desc` (`countrycode`, `id_parameter`, `fieldname`, `unit`, `description`) VALUES
('CH', 'tre200s0', 'temperature', '°C', 'Lufttemperatur 2 m über Boden; Momentanwert'),
('CH', 'sre000z0', 'sun_duration', 'minutes', 'Sonnenscheindauer; Zehnminutensumme'),
('CH', 'rre150z0', 'rain', 'mm', 'Niederschlag; Zehnminutensumme'),
('CH', 'dkl010z0', 'wind_direction', '°', 'Windrichtung; Zehnminutenmittel'),
('CH', 'fu3010z0', 'wind_speed', 'km/h', 'Windgeschwindigkeit; Zehnminutenmittel'),
('CH', 'pp0qnhs0', 'pressure_QNH', 'km/h', 'Luftdruck reduziert auf Meeresniveau mit Standardatmosphäre (QNH); Momentanwert'),
('CH', 'fu3010z1', 'wind_max', 'km/h', 'Böenspitze (Sekundenböe);Maximum'),
('CH', 'ure200s0', 'relative_humidity', '%', 'Relative Luftfeuchtigkeit2 m über Boden; Momentanwert'),
('CH', 'prestas0', 'pressure_QFE', 'hPa', 'Luftdruck auf Stationshöhe (QFE); Momentanwert'),
('CH', 'pp0qffs0', 'pressure_QFF', 'hPa', 'Luftdruck reduziert auf Meeresniveau (QFF); Momentanwert');

-- --------------------------------------------------------

--
-- Struttura della tabella `tbstation`
--

CREATE TABLE `tbmeteo_station` (
  `countrycode` varchar(5) NOT NULL,
  `id_station` varchar(16) NOT NULL,
  `name` varchar(128) DEFAULT NULL,
  `longlat` varchar(32) DEFAULT NULL,
  `km_coordinates` varchar(32) DEFAULT NULL,
  `altitude` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dump dei dati per la tabella `tbstation`
--

INSERT INTO `tbmeteo_station` (`countrycode`, `id_station`, `name`, `longlat`, `km_coordinates`, `altitude`) VALUES
('CH', 'ABO', 'Adelboden', '7°34/46°30', '609400/148975', 1320),
('CH', 'AIG', 'Aigle', '6°55/46°20', '560400/130713', 381),
('CH', 'ALT', 'Altdorf', '8°37/46°53', '690174/193558', 438),
('CH', 'AND', 'Andeer', '9°26/46°37', '752687/164035', 987),
('CH', 'ATT', 'Les Attelas', '7°16/46°06', '586862/105305', 2730),
('CH', 'BAS', 'Basel / Binningen', '7°35/47°32', '610911/265600', 316),
('CH', 'BER', 'Bern / Zollikofen', '7°28/46°59', '601929/204409', 552),
('CH', 'BEZ', 'Beznau', '8°14/47°33', '659808/267693', 325),
('CH', 'BIE', 'Bière', '6°21/46°31', '515888/153206', 683),
('CH', 'BIZ', 'Bischofszell', '9°14/47°30', '735325/262285', 470),
('CH', 'BOL', 'Boltigen', '7°23/46°37', '595828/163588', 820),
('CH', 'BOU', 'Bouveret', '6°51/46°24', '555264/138175', 374),
('CH', 'BRZ', 'Brienz', '8°04/46°44', '647546/176806', 567),
('CH', 'BUF', 'Buffalora', '10°16/46°39', '816494/170225', 1968),
('CH', 'BUS', 'Buchs / Aarau', '8°05/47°23', '648389/248365', 386),
('CH', 'CDF', 'La Chaux-de-Fonds', '6°48/47°05', '550923/214893', 1018),
('CH', 'CGI', 'Nyon / Changins', '6°14/46°24', '506880/139573', 455),
('CH', 'CHA', 'Chasseral', '7°03/47°08', '570842/220154', 1599),
('CH', 'CHD', 'Château-d`Oex', '7°08/46°29', '577041/147644', 1029),
('CH', 'CHU', 'Chur', '9°32/46°52', '759471/193157', 556),
('CH', 'CIM', 'Cimetta', '8°47/46°12', '704433/117452', 1661),
('CH', 'CMA', 'Crap Masegn', '9°11/46°51', '732820/189380', 2480),
('CH', 'COM', 'Acquarossa / Comprovasco', '8°56/46°28', '714998/146440', 575),
('CH', 'COV', 'Piz Corvatsch', '9°49/46°25', '783146/143519', 3305),
('CH', 'CRM', 'Cressier', '7°04/47°03', '571160/210800', 431),
('CH', 'DAV', 'Davos', '9°51/46°49', '783514/187457', 1594),
('CH', 'DEM', 'Delémont', '7°21/47°21', '593269/244543', 439),
('CH', 'DIS', 'Disentis / Sedrun', '8°51/46°42', '708188/173789', 1197),
('CH', 'DOL', 'La Dôle', '6°06/46°25', '497061/142362', 1669),
('CH', 'EBK', 'Ebnat-Kappel', '9°07/47°16', '726348/237167', 623),
('CH', 'EGH', 'Eggishorn', '8°06/46°26', '650279/141897', 2893),
('CH', 'EGO', 'Egolzwil', '8°00/47°11', '642910/225537', 521),
('CH', 'EIN', 'Einsiedeln', '8°45/47°08', '699981/221058', 910),
('CH', 'ELM', 'Elm', '9°11/46°55', '732265/198425', 958),
('CH', 'ENG', 'Engelberg', '8°25/46°49', '674156/186097', 1035),
('CH', 'EVO', 'Evolène / Villa', '7°31/46°07', '605415/106740', 1825),
('CH', 'FAH', 'Fahy', '6°56/47°25', '562458/252676', 596),
('CH', 'FRE', 'Bullet / La Frétaz', '6°35/46°50', '534221/188081', 1205),
('CH', 'GEN', 'Monte Generoso', '9°01/45°56', '722503/87456', 1600),
('CH', 'GIH', 'Giswil', '8°11/46°51', '657320/188980', 475),
('CH', 'GLA', 'Glarus', '9°04/47°02', '723752/210567', 516),
('CH', 'GOE', 'Gösgen', '7°58/47°22', '640417/245937', 380),
('CH', 'GOR', 'Gornergrat', '7°47/45°59', '626900/92512', 3129),
('CH', 'GRA', 'Fribourg / Posieux', '7°07/46°46', '575182/180076', 646),
('CH', 'GRE', 'Grenchen', '7°25/47°11', '598216/225348', 430),
('CH', 'GRH', 'Grimsel Hospiz', '8°20/46°34', '668583/158215', 1980),
('CH', 'GRO', 'Grono', '9°10/46°15', '733014/124080', 323),
('CH', 'GSB', 'Col du Grand St-Bernard', '7°10/45°52', '579200/79720', 2472),
('CH', 'GUE', 'Gütsch ob Andermatt', '8°37/46°39', '690140/167590', 2287),
('CH', 'GUT', 'Güttingen', '9°17/47°36', '738419/273960', 440),
('CH', 'GVE', 'Genève-Cointrin', '6°08/46°15', '498903/122624', 420),
('CH', 'HAI', 'Salen-Reutenen', '9°01/47°39', '719102/279042', 718),
('CH', 'HLL', 'Hallau', '8°28/47°42', '677456/283472', 419),
('CH', 'HOE', 'Hörnli', '8°56/47°22', '713515/247755', 1132),
('CH', 'INT', 'Interlaken', '7°52/46°40', '633019/169093', 577),
('CH', 'JUN', 'Jungfraujoch', '7°59/46°33', '641930/155275', 3580),
('CH', 'KLO', 'Zürich / Kloten', '8°32/47°29', '682706/259337', 426),
('CH', 'KOP', 'Koppigen', '7°36/47°07', '612662/218664', 484),
('CH', 'LAE', 'Lägern', '8°24/47°29', '672250/259460', 845),
('CH', 'LAG', 'Langnau i.E.', '7°48/46°56', '628005/198792', 745),
('CH', 'LEI', 'Leibstadt', '8°11/47°36', '656378/272111', 341),
('CH', 'LUG', 'Lugano', '8°58/46°00', '717873/95884', 273),
('CH', 'LUZ', 'Luzern', '8°18/47°02', '665540/209848', 454),
('CH', 'MAG', 'Magadino / Cadenazzo', '8°56/46°10', '715475/113162', 203),
('CH', 'MER', 'Meiringen', '8°10/46°44', '655843/175920', 588),
('CH', 'MLS', 'Le Moléson', '7°01/46°33', '567723/155072', 1974),
('CH', 'MOA', 'Mosen', '8°14/47°15', '660124/232846', 452),
('CH', 'MOE', 'Möhlin', '7°53/47°34', '633050/269142', 344),
('CH', 'MRP', 'Monte Rosa-Plattje', '7°49/45°57', '629149/89520', 2885),
('CH', 'MUB', 'Mühleberg', '7°17/46°58', '587788/202478', 479),
('CH', 'MVE', 'Montana', '7°28/46°18', '601706/127482', 1427),
('CH', 'NAP', 'Napf', '7°56/47°00', '638132/206078', 1403),
('CH', 'NAS', 'Naluns / Schlivera', '10°16/46°49', '815374/188987', 2400),
('CH', 'NEU', 'Neuchâtel', '6°57/47°00', '563150/205600', 485),
('CH', 'ORO', 'Oron', '6°51/46°34', '555502/158048', 827),
('CH', 'OTL', 'Locarno / Monti', '8°47/46°10', '704160/114350', 366),
('CH', 'PAY', 'Payerne', '6°57/46°49', '562127/184612', 490),
('CH', 'PIL', 'Pilatus', '8°15/46°59', '661910/203410', 2106),
('CH', 'PIO', 'Piotta', '8°41/46°31', '695888/152261', 990),
('CH', 'PLF', 'Plaffeien', '7°16/46°45', '586808/177400', 1042),
('CH', 'PMA', 'Piz Martegnas', '9°32/46°35', '760267/160583', 2670),
('CH', 'PRE', 'St-Prex', '6°27/46°29', '523549/148525', 425),
('CH', 'PUY', 'Pully', '6°40/46°31', '540811/151514', 455),
('CH', 'QUI', 'Quinten', '9°13/47°08', '734848/221278', 419),
('CH', 'RAG', 'Bad Ragaz', '9°30/47°01', '756907/209340', 496),
('CH', 'REH', 'Zürich / Affoltern', '8°31/47°26', '681428/253545', 443),
('CH', 'ROB', 'Poschiavo / Robbia', '10°04/46°21', '801850/136180', 1078),
('CH', 'ROE', 'Robièi', '8°31/46°27', '682587/144091', 1894),
('CH', 'RUE', 'Rünenberg', '7°53/47°26', '633246/253845', 611),
('CH', 'SAE', 'Säntis', '9°21/47°15', '744200/234920', 2502),
('CH', 'SAM', 'Samedan', '9°53/46°32', '787210/155700', 1708),
('CH', 'SBE', 'S. Bernardino', '9°11/46°28', '734112/147296', 1638),
('CH', 'SBO', 'Stabio', '8°56/45°51', '716034/77964', 353),
('CH', 'SCM', 'Schmerikon', '8°56/47°13', '713722/231496', 408),
('CH', 'SCU', 'Scuol', '10°17/46°48', '817135/186393', 1303),
('CH', 'SHA', 'Schaffhausen', '8°37/47°41', '688698/282796', 438),
('CH', 'SIO', 'Sion', '7°20/46°13', '591630/118575', 482),
('CH', 'SMA', 'Zürich / Fluntern', '8°34/47°23', '685117/248061', 555),
('CH', 'SMM', 'Sta. Maria, Val Müstair', '10°26/46°36', '828858/165569', 1383),
('CH', 'SPF', 'Schüpfheim', '8°01/46°57', '643677/199706', 742),
('CH', 'STG', 'St. Gallen', '9°24/47°26', '747861/254586', 775),
('CH', 'STK', 'Steckborn', '8°59/47°40', '715871/280916', 398),
('CH', 'TAE', 'Aadorf / Tänikon', '8°54/47°29', '710514/259821', 539),
('CH', 'THU', 'Thun', '7°35/46°45', '611202/177630', 570),
('CH', 'TIT', 'Titlis', '8°26/46°46', '675400/180400', 3040),
('CH', 'ULR', 'Ulrichen', '8°18/46°30', '666740/150760', 1345),
('CH', 'VAB', 'Valbella', '9°33/46°45', '761637/180380', 1569),
('CH', 'VAD', 'Vaduz', '9°31/47°08', '757718/221696', 457),
('CH', 'VIS', 'Visp', '7°51/46°18', '631149/128020', 639),
('CH', 'WAE', 'Wädenswil', '8°41/47°13', '693849/230708', 485),
('CH', 'WFJ', 'Weissfluhjoch', '9°48/46°50', '780615/189635', 2690),
('CH', 'WYN', 'Wynau', '7°47/47°15', '626400/233850', 422),
('CH', 'ZER', 'Zermatt', '7°45/46°02', '624350/97566', 1638);

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
