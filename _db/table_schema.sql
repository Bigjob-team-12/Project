DROP TABLE IF EXISTS url;
CREATE TABLE `url` (
	`no` INT NOT NULL AUTO_INCREMENT,
	`url_num` INT NOT NULL DEFAULT '0' COMMENT 'url num(2,3)',
	`num` INT NOT NULL DEFAULT '0' COMMENT '고유번호',
	`image` MEDIUMTEXT NOT NULL COMMENT '이미지' COLLATE 'utf8mb4_0900_ai_ci',
	PRIMARY KEY (`no`) USING BTREE
)
COLLATE='utf8mb4_0900_ai_ci'
ENGINE=InnoDB
;


DROP TABLE IF EXISTS protect_animals_url1;
CREATE TABLE `protect_animals_url1` (
	`no` INT NOT NULL AUTO_INCREMENT,
	`number` VARCHAR(25) NOT NULL DEFAULT '' COMMENT '공고번호' COLLATE 'utf8mb4_0900_ai_ci',
	`kind` VARCHAR(20) NOT NULL DEFAULT '' COMMENT '품종' COLLATE 'utf8mb4_0900_ai_ci',
	`color` VARCHAR(30) NOT NULL DEFAULT '' COMMENT '색상' COLLATE 'utf8mb4_0900_ai_ci',
	`sex` VARCHAR(10) NOT NULL DEFAULT '' COMMENT '성별' COLLATE 'utf8mb4_0900_ai_ci',
	`neutralization` VARCHAR(10) NOT NULL DEFAULT '' COMMENT '중성화 여부' COLLATE 'utf8mb4_0900_ai_ci',
	`age_weight` VARCHAR(30) NOT NULL DEFAULT '' COMMENT '나이/체중' COLLATE 'utf8mb4_0900_ai_ci',
	`date` DATE NOT NULL DEFAULT '0000-00-00' COMMENT '접수일시',
	`location` VARCHAR(50) NOT NULL DEFAULT '0' COMMENT '발견 장소' COLLATE 'utf8mb4_0900_ai_ci',
	`characteristic` MEDIUMTEXT NOT NULL COMMENT '특징' COLLATE 'utf8mb4_0900_ai_ci',
	`deadline` CHAR(20) NOT NULL COMMENT '공고기한' COLLATE 'utf8mb4_0900_ai_ci',
	`center_name` VARCHAR(50) NOT NULL DEFAULT '0' COMMENT '보호센터 이름' COLLATE 'utf8mb4_0900_ai_ci',
	`center_number` VARCHAR(15) NOT NULL DEFAULT '0' COMMENT '보호센터 전화번호' COLLATE 'utf8mb4_0900_ai_ci',
	`center_address` VARCHAR(200) NOT NULL DEFAULT '0' COMMENT '보호센터 주소' COLLATE 'utf8mb4_0900_ai_ci',
	`image` VARCHAR(100) NOT NULL COMMENT '이미지' COLLATE 'utf8mb4_0900_ai_ci',
	`url` MEDIUMTEXT NOT NULL COMMENT 'scraping 한 url' COLLATE 'utf8mb4_0900_ai_ci',
	`time` DATETIME NOT NULL COMMENT 'scraping 한 시간',
	PRIMARY KEY (`no`) USING BTREE
)
COLLATE='utf8mb4_0900_ai_ci'
ENGINE=InnoDB
AUTO_INCREMENT=11
;


DROP TABLE IF EXISTS protect_animals_url2;
CREATE TABLE `protect_animals_url2` (
	`no` INT NOT NULL AUTO_INCREMENT,
	`number` INT NOT NULL DEFAULT '0' COMMENT '고유 번호',
	`name` VARCHAR(20) NOT NULL DEFAULT '' COMMENT '등록인' COLLATE 'utf8mb4_0900_ai_ci',
	`date` DATE NOT NULL COMMENT '발견한 날짜',
	`phone_num` CHAR(13) NOT NULL COMMENT '연락처' COLLATE 'utf8mb4_0900_ai_ci',
	`sex` VARCHAR(10) NOT NULL COMMENT '성별' COLLATE 'utf8mb4_0900_ai_ci',
	`address` VARCHAR(50) NOT NULL COMMENT '발견한 장소' COLLATE 'utf8mb4_0900_ai_ci',
	`title` MEDIUMTEXT NOT NULL COMMENT '글 제목' COLLATE 'utf8mb4_0900_ai_ci',
	`text` MEDIUMTEXT NOT NULL COMMENT '내용' COLLATE 'utf8mb4_0900_ai_ci',
	`url` MEDIUMTEXT NOT NULL COMMENT 'scraping 한 url' COLLATE 'utf8mb4_0900_ai_ci',
	`time` DATETIME NOT NULL COMMENT 'scraping 한 시간',
	PRIMARY KEY (`no`) USING BTREE
)
COLLATE='utf8mb4_0900_ai_ci'
ENGINE=InnoDB
AUTO_INCREMENT=11
;

DROP TABLE IF EXISTS missing_animals_url3;
CREATE TABLE `missing_animals_url3` (
	`no` INT NOT NULL AUTO_INCREMENT,
	`number` INT NOT NULL COMMENT '고유번호',
	`phone_num` CHAR(13) NOT NULL COMMENT '실종자 전화번호' COLLATE 'utf8mb4_0900_ai_ci',
	`address` VARCHAR(50) NOT NULL COMMENT '실종된 장소' COLLATE 'utf8mb4_0900_ai_ci',
	`date` VARCHAR(100) NOT NULL COMMENT '실종된 날짜' COLLATE 'utf8mb4_0900_ai_ci',
	`title` MEDIUMTEXT NOT NULL COMMENT '글 제목' COLLATE 'utf8mb4_0900_ai_ci',
	`sex` VARCHAR(10) NOT NULL COMMENT '성별' COLLATE 'utf8mb4_0900_ai_ci',
	`text` MEDIUMTEXT NOT NULL COMMENT '상세 설명' COLLATE 'utf8mb4_0900_ai_ci',
	`url` MEDIUMTEXT NOT NULL COMMENT 'scraping 한 url' COLLATE 'utf8mb4_0900_ai_ci',
	`time` DATETIME NOT NULL COMMENT 'scraping 한 time',
	PRIMARY KEY (`no`) USING BTREE
)
COLLATE='utf8mb4_0900_ai_ci'
ENGINE=InnoDB
AUTO_INCREMENT=30
;

