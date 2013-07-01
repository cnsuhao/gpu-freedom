-- phpMyAdmin SQL Dump
-- version 3.5.2
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Jul 01, 2013 at 04:10 PM
-- Server version: 5.5.25a
-- PHP Version: 5.4.4

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `powergrid`
--

-- --------------------------------------------------------

--
-- Table structure for table `frequency`
--

CREATE TABLE IF NOT EXISTS `frequency` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `frequencyhz` double NOT NULL,
  `networkdiff` double DEFAULT NULL,
  `controlarea` varchar(16) NOT NULL,
  `tso` varchar(16) NOT NULL,
  `create_dt` datetime NOT NULL,
  `create_user` varchar(16) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=20 ;

--
-- Dumping data for table `frequency`
--

INSERT INTO `frequency` (`id`, `frequencyhz`, `networkdiff`, `controlarea`, `tso`, `create_dt`, `create_user`) VALUES
(2, 50.007, 13.201, 'SG_ST', 'Swissgrid', '2013-06-28 15:23:06', 'php'),
(3, 50.013, 13.14, 'SG_ST', 'Swissgrid', '2013-06-28 15:23:56', 'php'),
(4, 50.01, 13.149, 'SG_ST', 'Swissgrid', '2013-06-28 15:24:22', 'php'),
(5, 50.01, 13.149, 'SG_ST', 'Swissgrid', '2013-06-28 15:30:40', 'php'),
(6, 49.996, 13.141, 'SG_ST', 'Swissgrid', '2013-06-28 15:46:31', 'php'),
(7, 49.985, 13.154, 'SG_ST', 'Swissgrid', '2013-06-28 16:06:21', 'php'),
(8, 49.985, 13.154, 'SG_ST', 'Swissgrid', '2013-06-28 16:06:44', 'php'),
(9, 49.998, 12.954, 'SG_ST', 'Swissgrid', '2013-06-28 16:24:11', 'php'),
(10, 50.01, 12.902, 'SG_ST', 'Swissgrid', '2013-06-28 17:02:20', 'php'),
(11, 50.011, 40.321, 'SG_ST', 'Swissgrid', '2013-07-01 08:03:46', 'php'),
(12, 49.977, 39.731, 'SG_ST', 'Swissgrid', '2013-07-01 08:52:27', 'php'),
(13, 49.973, 39.458, 'SG_ST', 'Swissgrid', '2013-07-01 09:27:46', 'php'),
(14, 50.011, 38.976, 'SG_ST', 'Swissgrid', '2013-07-01 10:02:25', 'php'),
(15, 50.002, 38.669, 'SG_ST', 'Swissgrid', '2013-07-01 10:42:35', 'php'),
(16, 49.967, 36.187, 'SG_ST', 'Swissgrid', '2013-07-01 15:53:17', 'php'),
(17, 49.981, 35.472, 'SG_ST', 'Swissgrid', '2013-07-01 15:59:30', 'php'),
(19, 50.006, 35.373, 'SG_ST', 'Swissgrid', '2013-07-01 16:09:12', 'php');

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
